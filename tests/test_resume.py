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
from oxyformer.provenance import ContractError, read_artifact
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
        load_checkpoint(replace(first, path=str(damaged), sha256=sha256(payload).hexdigest()), first.identity)
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
    artifact = replace(first, path=str(path), sha256=sha256(payload).hexdigest())
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
