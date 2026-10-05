"""CPU FP32 is mandatory. GPU audit is explicit and never enables BF16."""
from dataclasses import fields, replace
import os
from pathlib import Path

import pytest
import torch
import yaml

from test_nested_cv import full, threads, predictions, state
from oxyformer.training import nested_cv as nested
from oxyformer.validation.leakage import audit_precision


def test_cpu_fp32_parameters_and_predictions(full):
    prepared, artifact = full
    for bundle in state(artifact)["final"].values():
        assert all(v.dtype == torch.float32 for v in bundle["state"].values()
                   if isinstance(v, torch.Tensor) and v.is_floating_point())
    original = predictions(prepared, artifact)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            altered = predictions(prepared, artifact)
    finally:
        torch.set_default_dtype(previous)
    assert original.mu_a == altered.mu_a and original.r_a == altered.r_a
    assert all(isinstance(v, float) for v in original.mu_a + original.r_a)


def test_precision_measurements_and_bf16_disabled():
    reference = torch.tensor([0., 1., 2.], dtype=torch.float32)
    assert audit_precision(reference, reference.clone(), absolute_tolerance=1e-5, relative_tolerance=1e-4)["passed"]
    assert not audit_precision(reference, reference + .1, absolute_tolerance=1e-5, relative_tolerance=1e-4)["passed"]
    assert yaml.safe_load(Path("configs/training/nested.yaml").read_text())["bf16_enabled"] is False


@pytest.mark.skipif(os.environ.get("OXYFORMER_GPU_PRECISION_AUDIT") != "1" or not torch.cuda.is_available(),
                    reason="opt-in GPU precision audit requires OXYFORMER_GPU_PRECISION_AUDIT=1 and CUDA")
def test_gpu_encoder_precision_audit(full, tmp_path):
    prepared, artifact = full
    encoder = nested._build(state(artifact)["final"]["outcome"]).encoder.eval().cuda()
    view = prepared.data.covariates(("x",))
    batch = encoder.tokenizer.prepare(view)
    batch = replace(batch, **{f.name: getattr(batch, f.name).cuda() for f in fields(batch)
                             if isinstance(getattr(batch, f.name), torch.Tensor)})
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        fp32 = encoder(batch)[1]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = encoder(batch)[1]
    audit = audit_precision(fp32, bf16, absolute_tolerance=1e-2, relative_tolerance=1e-2)
    (tmp_path / "gpu-precision.json").write_text(nested.canonical_json(audit))
    assert audit["passed"], audit
    assert not audit["bf16_enabled"]
