"""Bit-exact interrupted optimization and strict permissible-data cache identity."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import pickle
import random
import signal

import numpy as np
import pytest
import torch

from test_pretrain import cpu_only, make_case
from oxyformer.provenance import ContractError, canonical_json, read_artifact
from oxyformer.training.checkpoint import (
    CheckpointArtifact, CheckpointRequest, capture_rng, load_checkpoint, restore_rng, save_checkpoint,
)
from oxyformer.training.pretrain import StatefulSampler, pretrain


def assert_state_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_state_equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("budget", [1, 4, 5, 7])
def test_interrupted_optimization_is_bit_exact(tmp_path, budget):
    view, split, config = make_case(tmp_path)
    controller = {"outer_fold": 0, "inner": {"candidate": 2, "completed": [0, 1]}, "shards": []}
    config = replace(config, controller_state=controller)
    full = pretrain(view, split, config, 1103)
    first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"), max_batches=budget), 1103)
    assert not first.complete and first.reason == "batch_limit"
    original_bytes = Path(first.path).read_bytes()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.rand(13)
    resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"), predecessor=first), 1103)
    assert resumed.complete and resumed.predecessor_hash == first.content_hash
    assert Path(first.path).read_bytes() == original_bytes
    assert Path(resumed.path).parent != Path(first.path).parent
    assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(resumed, resumed.identity))
    assert full.lineage.model_hash == resumed.lineage.model_hash


@pytest.mark.parametrize("seed", [0, 2**63, 2**64 - 1])
def test_seed_boundaries_resume_bit_exactly(tmp_path, seed):
    view, split, config = make_case(tmp_path)
    split = replace(split, seed_ids=(seed,))
    config = replace(config, settings=replace(config.settings, max_epochs=1))
    full = pretrain(view, split, config, seed)
    first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"),
                                        max_batches=1), seed)
    resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"),
                                          predecessor=first), seed)
    assert not first.complete and full.complete and resumed.complete
    assert full.identity.seed == resumed.identity.seed == seed
    assert full.lineage.seed == resumed.lineage.seed == seed
    assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(resumed, resumed.identity))


@pytest.mark.parametrize("seed", [-1, 2**64, 2**100, True])
def test_invalid_seed_is_rejected_before_attempt_or_rng_mutation(tmp_path, seed):
    view, split, config = make_case(tmp_path)
    if type(seed) is int and seed >= 0:
        # The merged manifest's structural contract permits oversized integers.
        split = replace(split, seed_ids=(seed,))
    initial_rng = capture_rng()
    with pytest.raises(ContractError, match=r"SSL seed must be a Python integer in \[0, 2\*\*64 - 1\]"):
        pretrain(view, split, config, seed)
    assert not Path(config.output_dir).exists()
    assert_state_equal(initial_rng, capture_rng())


def test_many_tiny_slices_complete_without_restarting_epochs(tmp_path):
    view, split, config = make_case(tmp_path)
    full = pretrain(view, split, config, 1103)
    predecessor = None
    for attempt in range(16):
        predecessor = pretrain(view, split, replace(config, output_dir=str(tmp_path / f"slice-{attempt}"),
                              predecessor=predecessor, max_batches=1), 1103)
        if predecessor.complete:
            break
    assert attempt == 14 and predecessor.complete
    assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(predecessor, predecessor.identity))


def test_sampler_restores_current_order_cursor_and_future_permutations():
    uninterrupted = StatefulSampler(11, 1103)
    uninterrupted.indices(3)
    uninterrupted.cursor += 3
    interrupted = StatefulSampler(11, 999)
    interrupted.load_state_dict(uninterrupted.state_dict())
    for _ in range(3):
        assert uninterrupted.indices(3) == interrupted.indices(3)
        n = len(uninterrupted.indices(3))
        uninterrupted.cursor += n
        interrupted.cursor += n
    uninterrupted.finish_epoch()
    interrupted.finish_epoch()
    assert uninterrupted.indices(11) == interrupted.indices(11)


def test_rng_roundtrip_restores_python_numpy_and_torch():
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert_state_equal(expected, actual)


@pytest.mark.parametrize("change", ["values", "ids", "order", "split", "config", "seed", "code", "environment", "preprocessing"])
def test_reuse_requires_identical_scientific_identity(tmp_path, monkeypatch, change):
    import oxyformer.training.pretrain as module
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    resumed_config = replace(config, predecessor=first, output_dir=str(tmp_path / "resume"))
    seed = 1103
    if change == "values":
        view = replace(view, values=((99., 1., "rent"),) + view.values[1:])
    elif change == "ids":
        ids = ("new-id",) + view.original_ids[1:]
        view = replace(view, original_ids=ids, lineage=replace(view.lineage, unit_ids=ids))
    elif change == "order":
        view = replace(view, original_ids=view.original_ids[::-1], values=view.values[::-1])
    elif change == "split":
        split = replace(split, level="outer")
    elif change == "config":
        resumed_config = replace(resumed_config, settings=replace(config.settings, learning_rate=1e-3))
    elif change == "seed":
        seed = 2207
    elif change == "code":
        monkeypatch.setattr(module, "scientific_code_fingerprint", lambda: "b" * 64)
    elif change == "environment":
        monkeypatch.setattr(module, "environment_identity", lambda device: (("changed", "environment"),))
    elif change == "preprocessing":
        original = module.fit_preprocessing
        monkeypatch.setattr(module, "fit_preprocessing", lambda *a: (replace(original(*a)[0], mean=99),) + original(*a)[1:])
    with pytest.raises(ContractError, match="identity|permitted training"):
        pretrain(view, split, resumed_config, seed)
    assert not Path(resumed_config.output_dir).exists()


def test_stale_preprocessing_inside_valid_archive_is_rejected(tmp_path):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    state = load_checkpoint(first, first.identity)
    state["preprocessing"] = ("stale",) + state["preprocessing"][1:]
    root = tmp_path / "stale"
    root.mkdir()
    stale = save_checkpoint(root, identity=first.identity, lineage=first.lineage, state=state,
                            complete=False, reason="batch_limit")
    with pytest.raises(ContractError, match="stale preprocessing"):
        pretrain(view, split, replace(config, predecessor=stale, output_dir=str(tmp_path / "resume")), 1103)


def test_changed_controller_and_shared_writable_directory_are_refused(tmp_path):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1, controller_state={"fold": 1}), 1103)
    with pytest.raises(ContractError, match="controller"):
        pretrain(view, split, replace(config, predecessor=first, controller_state={"fold": 2}), 1103)
    with pytest.raises(FileExistsError):
        pretrain(view, split, replace(config, predecessor=first), 1103)


def test_completed_reuse_is_validated_and_copied_to_own_attempt(tmp_path):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, config, 1103)
    descriptor = read_artifact(Path(first.path).with_suffix(".json"), CheckpointArtifact, first.content_hash)
    copy = pretrain(view, split, replace(config, predecessor=descriptor, output_dir=str(tmp_path / "copy")), 1103)
    assert copy.complete and copy.step == first.step
    assert_state_equal(load_checkpoint(first, first.identity), load_checkpoint(copy, copy.identity))


def test_signal_handler_requests_checkpoint_without_io_and_restores_handler(tmp_path):
    view, split, config = make_case(tmp_path)
    request = CheckpointRequest()
    old_handler = signal.getsignal(signal.SIGUSR1)
    with request.signals():
        signal.raise_signal(signal.SIGUSR1)
        assert request.requested and not Path(config.output_dir).exists()
    assert signal.getsignal(signal.SIGUSR1) == old_handler
    first = pretrain(view, split, replace(config, stop_request=request), 1103)
    assert not first.complete and first.reason == "requested" and first.step == 0
    full = pretrain(view, split, replace(config, output_dir=str(tmp_path / "full")), 1103)
    resumed = pretrain(view, split, replace(config, predecessor=first, output_dir=str(tmp_path / "resume")), 1103)
    assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(resumed, resumed.identity))


def test_execution_deadline_does_not_claim_scientific_completion(tmp_path, monkeypatch):
    import oxyformer.training.pretrain as module
    view, split, config = make_case(tmp_path)
    ticks = iter([0., 14300.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    first = pretrain(view, split, config, 1103)
    assert not first.complete and first.reason == "slice_limit" and first.epoch == 0


def test_corrupted_archive_and_pickle_are_rejected_without_execution(tmp_path):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    damaged = tmp_path / "damaged.ofc"
    damaged.write_bytes(Path(first.path).read_bytes()[:-8])
    with pytest.raises(ContractError, match="hash mismatch"):
        load_checkpoint(replace(first, path=str(damaged)), first.identity)
    marker = tmp_path / "unsafe-executed"
    class Unsafe:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)
    payload = pickle.dumps(Unsafe())
    damaged.write_bytes(payload)
    with pytest.raises(ContractError, match="untrusted checkpoint serialization"):
        load_checkpoint(replace(first, path=str(damaged), sha256=sha256(payload).hexdigest(), byte_size=len(payload)), first.identity)
    assert not marker.exists()


def test_atomic_archive_publication_failure_has_no_descriptor(tmp_path, monkeypatch):
    import oxyformer.training.checkpoint as module
    view, split, config = make_case(tmp_path)
    def fail_link(*args):
        raise OSError("synthetic publication interruption")
    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="publication interruption"):
        pretrain(view, split, replace(config, max_batches=1), 1103)
    assert list((Path(config.output_dir) / "ssl").iterdir()) == []


def test_inconsistent_model_lineage_refused_on_publish(tmp_path):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    state = load_checkpoint(first, first.identity)
    key = next(iter(state["model"]))
    state["model"][key] = state["model"][key] + 1
    root = tmp_path / "substituted"
    root.mkdir()
    with pytest.raises(ContractError, match="model hash"):
        save_checkpoint(root, identity=first.identity, lineage=first.lineage, state=state,
                        complete=False, reason="batch_limit")
    assert list(root.iterdir()) == []


def test_inconsistent_model_lineage_refused_on_load(tmp_path):
    from io import BytesIO
    import zipfile
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    stream = BytesIO()
    with zipfile.ZipFile(first.path) as original, zipfile.ZipFile(stream, "w") as changed:
        for name in original.namelist():
            payload = original.read(name)
            if name == "tensors/0.npy":
                values = np.load(BytesIO(payload), allow_pickle=False)
                array = BytesIO()
                np.save(array, values + 1, allow_pickle=False)
                payload = array.getvalue()
            changed.writestr(name, payload)
    payload = stream.getvalue()
    path = tmp_path / "substituted.ofc"
    path.write_bytes(payload)
    artifact = replace(first, path=str(path), sha256=sha256(payload).hexdigest(), byte_size=len(payload))
    with pytest.raises(ContractError, match="model hash"):
        load_checkpoint(artifact, first.identity)


@pytest.fixture
def cuda_rng_on_cpu(monkeypatch):
    """Exercise real CUDA RNG wrappers with CPU generators, never a GPU."""
    generators = [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.random, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda.random, "_lazy_init", lambda: None)
    monkeypatch.setattr(torch.cuda.random, "_lazy_call", lambda fn, **kw: fn())
    monkeypatch.setattr(torch.cuda.random, "device_count", lambda: len(generators))
    monkeypatch.setattr(torch.cuda.random, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "default_generators", generators)
    return generators


def test_selected_cuda_rng_survives_removal_of_unused_gpu(cuda_rng_on_cpu):
    generators = cuda_rng_on_cpu
    state = capture_rng("cuda:0")
    expected = torch.rand(5, generator=generators[0])
    generators.pop()  # Same selected GPU; the other visible GPU disappears.
    restore_rng(state, "cuda:0")
    torch.testing.assert_close(torch.rand(5, generator=generators[0]), expected, rtol=0, atol=0)


def test_nondefault_cuda_rng_restores_only_selected_generator(cuda_rng_on_cpu):
    generators = cuda_rng_on_cpu
    state = capture_rng("cuda:1")
    expected = torch.rand(5, generator=generators[1])
    torch.rand(7, generator=generators[0])  # Unrelated work is not rewound.
    untouched = generators[0].get_state().clone()
    restore_rng(state, "cuda:1")
    torch.testing.assert_close(torch.rand(5, generator=generators[1]), expected, rtol=0, atol=0)
    assert torch.equal(generators[0].get_state(), untouched)


def test_cpu_rng_ignores_previously_initialized_cuda(cuda_rng_on_cpu, monkeypatch):
    state = capture_rng("cpu")
    assert state["cuda"] is None
    expected = torch.rand(5)
    cuda_rng_on_cpu.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restore_rng(state, "cpu")
    torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
    with pytest.raises(ContractError, match="RNG device mismatch"):
        restore_rng(state, "cuda:0")


@pytest.mark.parametrize("completed", [False, True])
def test_changed_default_dtype_rejects_resume_before_casting(tmp_path, completed):
    view, split, config = make_case(tmp_path)
    config = replace(config, settings=replace(config.settings, max_epochs=1))
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first = pretrain(view, split, replace(config, max_batches=None if completed else 1), 1103)
        predecessor_bytes = Path(first.path).read_bytes()
        torch.set_default_dtype(torch.float32)
        resume = replace(config, predecessor=first, output_dir=str(tmp_path / "resume"))
        before = capture_rng()
        with pytest.raises(ContractError, match="identity"):
            pretrain(view, split, resume, 1103)
        assert not Path(resume.output_dir).exists()
        assert_state_equal(before, capture_rng())
        assert Path(first.path).read_bytes() == predecessor_bytes
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("empty_partition", ["stopping", "fitting"])
def test_automatic_stopping_keeps_observed_targets_in_both_partitions(tmp_path, empty_partition):
    view, split, config = make_case(tmp_path)
    ranked = sorted(view.original_ids, key=lambda oid: sha256(
        canonical_json([1103, oid, "ssl-stopping"]).encode()).digest())
    missing = {ranked[0]} if empty_partition == "stopping" else set(ranked[2:])
    view = replace(view, values=tuple((None, None, None) if oid in missing else row
                                     for oid, row in zip(view.original_ids, view.values)))
    settings = replace(config.settings, stopping_ids=(), max_epochs=1,
                       validation_fraction=.1 if empty_partition == "stopping" else .9)
    config = replace(config, settings=settings)
    full = pretrain(view, split, config, 1103)
    state = load_checkpoint(full, full.identity)
    assert full.complete
    assert set(state["fitting_ids"]) - missing
    assert set(state["stopping_ids"]) - missing
    assert len(state["stopping_ids"]) == (1 if empty_partition == "stopping" else 9)
    assert set(state["fitting_ids"]) | set(state["stopping_ids"]) == set(split.training_ids(0))
    assert set(state["fitting_ids"]).isdisjoint(state["stopping_ids"])
    first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"), max_batches=1), 1103)
    resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"), predecessor=first), 1103)
    assert_state_equal(state, load_checkpoint(resumed, resumed.identity))


def test_compressed_checkpoint_is_refused_before_decompression(tmp_path, monkeypatch):
    from io import BytesIO
    import zipfile
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    stream = BytesIO()
    with zipfile.ZipFile(first.path) as original, zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED) as compressed:
        for name in original.namelist():
            compressed.writestr(name, original.read(name))
    payload = stream.getvalue()
    path = tmp_path / "compressed.ofc"
    path.write_bytes(payload)
    descriptor = replace(first, path=str(path), sha256=sha256(payload).hexdigest(), byte_size=len(payload))
    def forbid_decompression(*args, **kwargs):
        pytest.fail("checkpoint member was read before rejecting compression")
    monkeypatch.setattr(zipfile.ZipFile, "read", forbid_decompression)
    with pytest.raises(ContractError, match="compressed checkpoint"):
        load_checkpoint(descriptor, descriptor.identity)


@pytest.mark.parametrize("budget", [1, 4, 5])
@pytest.mark.parametrize("autocast_first", [False, True])
def test_ambient_autocast_cannot_change_exact_continuation(tmp_path, monkeypatch, budget, autocast_first):
    import oxyformer.training.pretrain as module
    constructor = module.MaskedReconstructor.__init__
    observed_modes = set()
    def observe_context(model, args):
        assert not torch.is_autocast_enabled("cpu") and not torch.is_inference_mode_enabled()
        assert torch.is_grad_enabled() == model.training
        observed_modes.add(model.training)
    def construct(self, *args):
        constructor(self, *args)
        self.register_forward_pre_hook(observe_context)
    monkeypatch.setattr(module.MaskedReconstructor, "__init__", construct)
    view, split, config = make_case(tmp_path)
    config = replace(config, settings=replace(config.settings, max_epochs=1))
    full = pretrain(view, split, config, 1103)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_first):
        first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"), max_batches=budget), 1103)
        assert torch.is_autocast_enabled("cpu") == autocast_first
    predecessor_bytes = Path(first.path).read_bytes()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not autocast_first):
        resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"), predecessor=first), 1103)
        assert torch.is_autocast_enabled("cpu") != autocast_first
    assert full.identity == first.identity == resumed.identity
    assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(resumed, resumed.identity))
    assert Path(first.path).read_bytes() == predecessor_bytes
    assert observed_modes == {False, True}


@pytest.mark.parametrize("ambient", ["no_grad", "inference", "placement"])
def test_training_owns_gradient_and_factory_contexts(tmp_path, ambient):
    view, split, config = make_case(tmp_path)
    config = replace(config, settings=replace(config.settings, max_epochs=1))
    full = pretrain(view, split, config, 1103)
    first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"), max_batches=1), 1103)
    context = {"no_grad": torch.no_grad, "inference": torch.inference_mode,
               "placement": lambda: torch.device("meta")}[ambient]
    with context():
        other = pretrain(view, split, replace(config, output_dir=str(tmp_path / "other")), 1103)
        resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"), predecessor=first), 1103)
        reused = pretrain(view, split, replace(config, output_dir=str(tmp_path / "reuse"), predecessor=full), 1103)
        if ambient == "no_grad":
            assert not torch.is_grad_enabled()
        elif ambient == "inference":
            assert torch.is_inference_mode_enabled()
        else:
            assert torch.empty(0).device.type == "meta"
    for artifact in (other, resumed, reused):
        assert_state_equal(load_checkpoint(full, full.identity), load_checkpoint(artifact, artifact.identity))


def test_execution_context_is_restored_after_training_exception(tmp_path, monkeypatch):
    import oxyformer.training.pretrain as module
    view, split, config = make_case(tmp_path)
    def fail_forward(*args):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert torch.empty(0).device.type == "cpu"
        raise RuntimeError("synthetic training failure")
    monkeypatch.setattr(module.MaskedReconstructor, "forward", fail_forward)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16), torch.device("meta"):
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            pretrain(view, split, config, 1103)
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert torch.is_autocast_enabled("cpu")
        assert torch.empty(0).device.type == "meta"


@pytest.mark.parametrize("control", ["mha_fastpath", "math_sdp_reduction"])
def test_changed_attention_policy_rejects_resume_before_attempt(tmp_path, control):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    getter, setter = {
        "mha_fastpath": (torch.backends.mha.get_fastpath_enabled, torch.backends.mha.set_fastpath_enabled),
        "math_sdp_reduction": (torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed,
                               torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp),
    }[control]
    previous = getter()
    try:
        setter(not previous)
        config = replace(config, predecessor=first, output_dir=str(tmp_path / "resume"))
        before = capture_rng()
        with pytest.raises(ContractError, match="identity"):
            pretrain(view, split, config, 1103)
        assert not Path(config.output_dir).exists()
        assert_state_equal(before, capture_rng())
    finally:
        setter(previous)


@pytest.mark.parametrize("failure", [None, "training", "validation", "publication"])
def test_caller_rng_is_restored_on_every_exit(tmp_path, monkeypatch, failure):
    import oxyformer.training.pretrain as module
    view, split, config = make_case(tmp_path)
    config = replace(config, settings=replace(config.settings, max_epochs=1))
    if failure in ("training", "validation"):
        forward = module.MaskedReconstructor.forward
        def maybe_fail(self, batch):
            if self.training == (failure == "training"):
                raise RuntimeError("synthetic execution failure")
            return forward(self, batch)
        monkeypatch.setattr(module.MaskedReconstructor, "forward", maybe_fail)
    elif failure == "publication":
        def fail_save(*args, **kwargs):
            raise RuntimeError("synthetic execution failure")
        monkeypatch.setattr(module, "save_checkpoint", fail_save)
    initial = capture_rng()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16), torch.device("meta"):
        if failure:
            with pytest.raises(RuntimeError, match="synthetic execution failure"):
                pretrain(view, split, config, 1103)
        else:
            assert pretrain(view, split, config, 1103).complete
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert torch.is_autocast_enabled("cpu") and torch.empty(0).device.type == "meta"
    assert_state_equal(initial, capture_rng())


def test_latched_request_stays_pending_and_fresh_request_allows_resume(tmp_path):
    view, split, config = make_case(tmp_path)
    request = CheckpointRequest()
    request.request()
    first = pretrain(view, split, replace(config, stop_request=request), 1103)
    repeated = pretrain(view, split, replace(config, output_dir=str(tmp_path / "pending"),
                                            predecessor=first, stop_request=request), 1103)
    assert first.reason == repeated.reason == "requested" and repeated.step == 0
    assert request.requested
    resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "fresh"),
                                           predecessor=repeated, stop_request=CheckpointRequest()), 1103)
    assert resumed.complete and resumed.step > repeated.step


def test_environment_identity_retains_unnamed_distributions_deterministically(monkeypatch):
    import importlib.metadata
    import oxyformer.training.pretrain as module
    from email.message import Message

    class Distribution:
        def __init__(self, name, version):
            self.metadata = Message()
            if name is not None:
                self.metadata["Name"] = name
            self.version = version

    distributions = [Distribution("named", "1"), Distribution(None, "2"),
                     Distribution(None, "1")]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(distributions))
    first = module.environment_identity(torch.device("cpu"))
    distributions.reverse()
    assert module.environment_identity(torch.device("cpu")) == first
    distributions[0].version = "3"
    assert module.environment_identity(torch.device("cpu")) != first
    distributions.pop(0)
    assert module.environment_identity(torch.device("cpu")) != first


@pytest.mark.parametrize("batch_size,budget", [(4, 1), (1, 7)])
def test_float64_large_loss_continues_through_validation(tmp_path, batch_size, budget):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        view, split, config = make_case(tmp_path)
        values = [0., 2e-308] * 3 + [1.] * 4
        view = replace(view, columns=("part", "complement"), values=tuple((v, v) for v in values))
        settings = replace(config.settings,
                           feature_kinds=(("part", "numeric"), ("complement", "numeric")),
                           families=(("part", "complement"),), batch_size=batch_size,
                           max_epochs=1, mask_rate=1.)
        config = replace(config, settings=settings)
        full = pretrain(view, split, config, 1103)
        state = load_checkpoint(full, full.identity)
        assert full.complete
        assert state["progress"]["history"] == pytest.approx([1e308])
        first = pretrain(view, split, replace(config, output_dir=str(tmp_path / "first"), max_batches=budget), 1103)
        resumed = pretrain(view, split, replace(config, output_dir=str(tmp_path / "resume"), predecessor=first), 1103)
        assert_state_equal(state, load_checkpoint(resumed, resumed.identity))
    finally:
        torch.set_default_dtype(previous)


def test_oversized_replacement_is_rejected_before_payload_read(tmp_path, monkeypatch):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    replaced = tmp_path / "oversized.ofc"
    with replaced.open("wb") as stream:
        stream.truncate(Path(first.path).stat().st_size + 32 * 1024 * 1024)
    original_open = Path.open

    class Guard:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size=-1):
            pytest.fail("oversized untrusted payload read before size/hash rejection")

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Guard(stream) if path == replaced else stream
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ContractError, match="size|hash"):
        load_checkpoint(replace(first, path=str(replaced)), first.identity)


def test_checkpoint_read_stays_bounded_if_file_grows_after_stat(tmp_path, monkeypatch):
    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    candidate = tmp_path / "growing.ofc"
    original_bytes = Path(first.path).read_bytes()
    candidate.write_bytes(original_bytes)
    original_open = Path.open

    class Guard:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size=-1):
            assert 0 <= size <= len(original_bytes) + 1, "unbounded read after stat"
            with original_open(candidate, "ab") as writer:
                writer.write(b"extra bytes")
            return self.stream.read(size)

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Guard(stream) if path == candidate else stream
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ContractError, match="size|hash"):
        load_checkpoint(replace(first, path=str(candidate)), first.identity)


@pytest.mark.parametrize("shape", [(100000000000000,), (2**32, 2**32)])
def test_npy_shape_is_bounded_before_array_allocation(tmp_path, monkeypatch, shape):
    from io import BytesIO
    import math
    import zipfile

    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    header = BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f8", "fortran_order": False, "shape": shape})
    archive = BytesIO()
    with zipfile.ZipFile(first.path) as source, zipfile.ZipFile(archive, "w") as output:
        for name in source.namelist():
            output.writestr(name, header.getvalue() if name == "tensors/0.npy" else source.read(name))
    data = archive.getvalue()
    path = tmp_path / "oversized-header.ofc"
    path.write_bytes(data)
    descriptor = replace(first, path=str(path), sha256=sha256(data).hexdigest(), byte_size=len(data))
    original_array = np.ndarray

    def bounded_array(shape, *args, **kwargs):
        count = math.prod(shape) if isinstance(shape, tuple) else int(shape)
        assert count <= len(data), "NPY header requested an allocation beyond trusted archive bytes"
        return original_array(shape, *args, **kwargs)

    monkeypatch.setattr(np, "ndarray", bounded_array)
    with pytest.raises(ContractError, match="tensor.*size"):
        load_checkpoint(descriptor, descriptor.identity)


@pytest.mark.parametrize("field,value", [
    ("unit_ids", ("other",)), ("split_hash", "b" * 64),
    ("config_hash", "c" * 64), ("environment", (("other", "environment"),)),
    ("seed", 999),
])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_lineage_must_agree_with_identity(tmp_path, field, value, operation):
    from io import BytesIO
    import json
    import zipfile

    view, split, config = make_case(tmp_path)
    first = pretrain(view, split, replace(config, max_batches=1), 1103)
    state = load_checkpoint(first, first.identity)
    lineage = replace(first.lineage, **{field: value})
    root = tmp_path / "contradictory"
    root.mkdir()
    if operation == "save":
        with pytest.raises(ContractError, match="lineage.*identity"):
            save_checkpoint(root, identity=first.identity, lineage=lineage, state=state,
                            complete=False, reason="batch_limit")
        assert list(root.iterdir()) == []
    else:
        stream = BytesIO()
        with zipfile.ZipFile(first.path) as source, zipfile.ZipFile(stream, "w") as output:
            for name in source.namelist():
                payload = source.read(name)
                if name == "metadata.json":
                    metadata = json.loads(payload)
                    metadata["lineage"] = lineage.to_dict()
                    payload = canonical_json(metadata).encode()
                output.writestr(name, payload)
        payload = stream.getvalue()
        path = root / "inconsistent.ofc"
        path.write_bytes(payload)
        descriptor = replace(first, path=str(path), lineage=lineage,
                             sha256=sha256(payload).hexdigest(), byte_size=len(payload))
        with pytest.raises(ContractError, match="lineage.*identity"):
            load_checkpoint(descriptor, first.identity)


@pytest.mark.parametrize("missing_rows", [0, 4])
def test_sparse_epoch_has_optimizer_update_and_exact_continuation(tmp_path, missing_rows):
    view, split, config = make_case(tmp_path)
    ids = view.original_ids[:missing_rows + 2]
    values = ((0.,),) + ((None,),) * missing_rows + ((1.,),)
    view = replace(view, original_ids=ids, columns=("part",), values=values,
                   lineage=replace(view.lineage, unit_ids=ids))
    all_ids = ids + ("external0", "external1")
    split = replace(split, original_ids=all_ids, fold_ids=(1,) * len(ids) + (0, 0),
                    lineage=replace(split.lineage, unit_ids=all_ids + split.design_ids + split.excluded_ids))
    settings = replace(config.settings, feature_kinds=(("part", "numeric"),), families=(("part",),),
                       stopping_ids=(ids[-1],), mask_rate=.3 if missing_rows == 0 else 1e-30,
                       batch_size=1, max_epochs=1)
    config = replace(config, settings=settings)
    full = pretrain(view, split, config, 1103)
    state = load_checkpoint(full, full.identity)
    assert full.complete
    assert state["optimizer"]["state"], "completed without an optimizer update"
    assert state["progress"]["epoch_updates"] >= 1
    first = pretrain(view, split, replace(config, max_batches=1,
                                         output_dir=str(tmp_path / "first")), 1103)
    resumed = pretrain(view, split, replace(config, predecessor=first,
                                           output_dir=str(tmp_path / "resumed")), 1103)
    assert_state_equal(state, load_checkpoint(resumed, resumed.identity))


def test_changed_blas_preference_rejects_resume_before_attempt(tmp_path):
    # A CUDA build can select cuBLASLt without a GPU, but a CPU-only wheel
    # cannot. Probe the actual build capability, not GPU visibility.
    preference = torch.backends.cuda.preferred_blas_library
    original = preference()
    try:
        try:
            preference("cublaslt")
        except RuntimeError as error:
            if "not been compiled with cuBLASLt" not in str(error):
                raise
            pytest.skip(f"PyTorch {torch.__version__} lacks cuBLASLt: {error}")
        preference("cublas")
        view, split, config = make_case(tmp_path)
        first = pretrain(view, split, replace(config, max_batches=1), 1103)
        preference("cublaslt")
        assert dict(first.identity.environment)["blas_preference"] != str(preference())
        config = replace(config, predecessor=first, output_dir=str(tmp_path / "resumed"))
        with pytest.raises(ContractError, match="identity"):
            pretrain(view, split, config, 1103)
        assert not Path(config.output_dir).exists()
    finally:
        if preference() != original:
            preference(original)
        assert preference() == original


def test_blas_preference_is_read_only_through_checkpoint_and_resume(tmp_path, monkeypatch):
    # Runs on both wheels, including when the cuBLASLt transition above skips.
    # Use the real getter, and fail if production ever tries to set a backend.
    preference = torch.backends.cuda.preferred_blas_library
    original = preference()
    reads = []

    def read_only(backend=None):
        assert backend is None, "CPU training must not set a CUDA BLAS backend"
        value = preference()
        reads.append(value)
        return value

    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", read_only)
    view, split, config = make_case(tmp_path)
    full = pretrain(view, split, config, 1103)
    first = pretrain(view, split, replace(config, max_batches=1,
                                         output_dir=str(tmp_path / "first")), 1103)
    descriptor = read_artifact(Path(first.path).with_suffix(".json"),
                               CheckpointArtifact, first.content_hash)
    assert descriptor == first
    assert dict(descriptor.identity.environment)["blas_preference"] == str(original)
    resumed = pretrain(view, split, replace(config, predecessor=descriptor,
                                           output_dir=str(tmp_path / "resumed")), 1103)
    assert resumed.complete and resumed.identity == full.identity
    assert_state_equal(load_checkpoint(full, full.identity),
                       load_checkpoint(resumed, resumed.identity))
    assert reads and all(value == original for value in reads)
    assert preference() == original
