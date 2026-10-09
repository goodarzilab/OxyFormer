"""Synthetic CPU/CUDA parity and deterministic continuation; no network/data.

CUDA tests skip explicitly with CPU-only wheels or hidden devices. Run them via
srun --gres=gpu:1. FP32 tolerances cover accumulated rounding across the small
nested network; reductions use FP64 but cannot recover FP32 forward precision.
No estimator, grid, seed, stopping schedule or scientific gate is replaced.
"""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from oxyformer.provenance import ContractError
from oxyformer.training import fit, nested_cv
from oxyformer.training.checkpoint import CheckpointArtifact, capture_rng
from test_nested_cv import endpoint, tiny_config, predictions, state

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: CPU-only wheel or no visible GPU")


def test_explicit_cpu_and_environment_selection(monkeypatch):
    monkeypatch.setenv("OXYFORMER_DEVICE", "cpu")
    assert fit.resolve_device().type == "cpu"
    assert fit.resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("OXYFORMER_DEVICE")
    assert fit.resolve_device().type == "cpu"
    with pytest.raises(ContractError, match="unavailable"):
        fit.resolve_device("cuda")


def test_cpu_scope_retains_reference_rng_and_arithmetic():
    before = capture_rng()
    with fit._numerics("cpu") as device:
        assert device.type == "cpu"
        assert torch.ones(1).device.type == "cpu"
        assert torch.ones(1).dtype == torch.float32
        torch.rand(10)
    assert torch.equal(capture_rng()["torch"], before["torch"])
    assert dict(fit.fit_environment("cpu"))["device"] == "cpu"


@GPU
def test_cuda_scope_uses_reference_dropout_masks_and_restores_policy():
    enabled = torch.are_deterministic_algorithms_enabled()
    precision = torch.backends.cuda.matmul.fp32_precision
    before = capture_rng("cuda")
    with fit._numerics("cuda"):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        torch.manual_seed(1103)
        cpu = torch.nn.functional.dropout(torch.ones(17, 64), .1)
        torch.manual_seed(1103)
        gpu = torch.nn.functional.dropout(torch.ones(17, 64, device="cuda"), .1)
        torch.testing.assert_close(cpu, gpu.cpu(), atol=0, rtol=0)
    assert torch.are_deterministic_algorithms_enabled() == enabled
    assert torch.backends.cuda.matmul.fp32_precision == precision
    assert torch.equal(capture_rng("cuda")["torch"], before["torch"])
    assert torch.equal(capture_rng("cuda")["cuda"], before["cuda"])


@pytest.fixture(scope="module")
def cuda_nested(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: CPU-only wheel or no visible GPU")
    root = tmp_path_factory.mktemp("gpu-nested")
    prepared = endpoint(root)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fitted = {}
        for device in ("cpu", "cuda"):
            config = tiny_config(prepared, root / device, device=device)
            seen = set()
            original_step = torch.optim.AdamW.step
            def observed_step(optimizer, *args, **kwargs):
                seen.update(parameter.device.type for group in optimizer.param_groups
                            for parameter in group["params"])
                return original_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.AdamW, "step", observed_step):
                fitted[device] = nested_cv.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
            assert fitted[device].complete
            assert seen == {device}, "Every SSL/nuisance AdamW update must use the requested device"
            for initialization in state(fitted[device])["initializations"].values():
                identity = CheckpointArtifact.from_json(initialization).identity
                assert dict(identity.environment)["device"].startswith(device)
        yield prepared, fitted
    finally:
        torch.set_num_threads(old)


@GPU
def test_small_nested_cpu_cuda_agree(cuda_nested):
    prepared, artifacts = cuda_nested
    cpu, gpu = (predictions(prepared, artifacts[k]) for k in ("cpu", "cuda"))
    assert state(artifacts["cpu"])["selection"] == state(artifacts["cuda"])["selection"]
    # 512 FP32 eps allows accumulated network/optimizer/calibration rounding,
    # while remaining far below differences from a changed stochastic stream.
    tolerance = 512 * torch.finfo(torch.float32).eps
    for name in ("mu_a", "mu_d", "r_a", "r_d"):
        torch.testing.assert_close(torch.tensor(getattr(cpu, name), dtype=torch.float64),
                                   torch.tensor(getattr(gpu, name), dtype=torch.float64),
                                   atol=tolerance, rtol=tolerance)
    assert dict(artifacts["cpu"].checkpoint.identity.environment)["device"] == "cpu"
    assert dict(artifacts["cuda"].checkpoint.identity.environment)["device"].startswith("cuda:")
    compute = state(artifacts["cuda"])["compute"]
    assert compute["gpu_seconds"] > 0 and compute["device"].startswith("cuda:")
    assert state(artifacts["cpu"])["compute"]["gpu_seconds"] == 0


@GPU
def test_cuda_continuation_is_exact_and_cpu_resume_is_refused(cuda_nested, tmp_path):
    prepared, artifacts = cuda_nested
    config = tiny_config(prepared, tmp_path / "partial", device="cuda", max_batches=15)
    partial = nested_cv.run_fold(config, prepared.outer, 1103, geography=prepared.geography)
    assert not partial.complete
    resumed = nested_cv.run_fold(replace(config, output_dir=str(tmp_path / "resume"),
        predecessor=partial, max_batches=None), prepared.outer, 1103, geography=prepared.geography)
    assert resumed.complete
    expected = artifacts["cuda"]
    assert resumed.checkpoint.lineage.model_hash == expected.checkpoint.lineage.model_hash
    assert state(resumed)["selection"] == state(expected)["selection"]
    for key in ("mu_a", "mu_d", "r_a", "r_d"):
        assert getattr(predictions(prepared, resumed), key) == getattr(predictions(prepared, expected), key)
    with pytest.raises(ContractError, match="identity"):
        nested_cv.run_fold(replace(config, device="cpu", output_dir=str(tmp_path / "wrong-device"),
            predecessor=partial, max_batches=None), prepared.outer, 1103, geography=prepared.geography)


def test_forced_tf32_is_refused_before_cuda_execution(monkeypatch):
    # The documented override bypasses PyTorch's FP32 precision setting.
    # Verify refusal before CUDA state is entered, including on CPU-only CI.
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setattr(fit, "resolve_device", lambda device: torch.device("cuda:0"))
    def forbidden_device(device):
        pytest.fail("CUDA execution reached with forced TF32")
    monkeypatch.setattr(torch.cuda, "device", forbidden_device)
    with pytest.raises(ContractError, match="TF32"):
        with fit._numerics("cuda"):
            pytest.fail("forced TF32 was admitted")


def test_visible_gpu_does_not_change_default(monkeypatch):
    monkeypatch.delenv("OXYFORMER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fit.resolve_device().type == "cpu"
    assert fit.resolve_device(fit.FitConfig.__dataclass_fields__["device"].default).type == "cpu"


@GPU
def test_cuda_profile_probe_task_path_on_registered_smoke_inputs(tmp_path, monkeypatch):
    """Exercise the probe wrapper and all fits with the registered smoke workload.

    The task keeps its production recipe; only this test's fitting boundary uses
    the registered synthetic one-epoch settings. This is execution coverage,
    not a production timing measurement or an alteration of the probe recipe.
    """
    import json
    from pathlib import Path
    from oxyformer.validation import coverage, real_frame, smoke_inputs
    from oxyformer.execution.runner import read_mapping
    from test_campaign import request, write

    monkeypatch.setenv('OXYFORMER_DEVICE', 'cuda')
    prepared, frame = smoke_inputs.build_inputs()
    root = Path(__file__).parents[1]
    smoke = read_mapping(root / 'configs/execution/tasks/campaign.yaml')['tasks'][1]['parameters']
    assert prepared.content_hash == smoke['recipe']['endpoint_hash']
    assert frame.content_hash == smoke['recipe']['frame_hash']
    task = real_frame.profile_tasks(prepared, frame)['tasks'][-1]
    assert task['id'] == 'profile-null-effect-gpu'
    inputs = tmp_path / 'inputs'
    write(inputs / 'endpoint.json', prepared.to_dict())
    write(inputs / 'frame.json', frame.to_dict())
    req = request(tmp_path / 'probe', task, {'real-frame-inputs': inputs})
    settings = nested_cv._settings
    monkeypatch.setattr(nested_cv, '_settings', lambda recipe, endpoint:
                        settings(smoke['recipe'], endpoint))
    run_fold = nested_cv.run_fold
    seen, environments = [], []
    def observed_fold(config, outer, seed, **kwargs):
        artifact = run_fold(config, outer, seed, **kwargs)
        assert artifact.complete
        environment = dict(artifact.checkpoint.identity.environment)
        assert environment['device'].startswith('cuda:')
        environments.append(environment)
        seen.append((config.fold, seed))
        return artifact
    monkeypatch.setattr(nested_cv, 'run_fold', observed_fold)
    old = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        result = coverage.run_stage(req)
    finally:
        torch.set_num_threads(old)
    assert result.status == 'pass', result.message
    result.verify(req)
    assert seen == [(fold, seed) for fold in range(5) for seed in nested_cv.SEEDS]
    timing = json.loads((Path(req.output_dir) / 'timing.json').read_text())
    assert timing['device'].startswith('cuda:')
    assert timing['gpu_seconds'] == timing['wall_seconds'] > 0
    assert timing['complete'] and timing['all_successful'] and not timing['budget_exceeded']
    assert all(timing['environment'] == environment for environment in environments)
    assert len(timing['complete_repetition_seconds']) == 1
    aggregate = json.loads((Path(req.output_dir) / 'result.json').read_text())
    assert aggregate['draws'] == task['parameters']['draws']
    assert aggregate['mode'] == 'profile'
    assert not aggregate['certifies_production_coverage']
