"""Tiny synthetic tensor checks; no data, GPU, downloads or network calls."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.tokens import FeatureBatch, FeatureSpec, FeatureTokenizer
from oxyformer.provenance import ContractError


@pytest.fixture(autouse=True)
def tiny_cpu_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1103)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def specs():
    return (FeatureSpec(name="income", kind="numeric", mean=10, scale=2),
            FeatureSpec(name="tenure", kind="categorical", categories=("own", "rent")),
            FeatureSpec(name="size", kind="numeric"))


@pytest.fixture
def batch():
    flags = torch.zeros((2, 3), dtype=torch.bool)
    return FeatureBatch(torch.arange(3), torch.tensor([[1., 0., -1.], [2., 0., 3.]]),
                        torch.tensor([[0, 0, 0], [0, 1, 0]]),
                        flags.clone(), flags.clone(), flags.clone())


def permute(batch, order):
    return FeatureBatch(batch.feature_ids[order], *(x[:, order] for x in (
        batch.numeric_values, batch.categorical_values, batch.missing, batch.masked, batch.padding)))


def test_numeric_identity_projection_and_missing_state(specs, batch):
    tokenizer = FeatureTokenizer(specs)
    with torch.no_grad():
        tokenizer.identity.weight.fill_(1)
        tokenizer.numeric_weight.fill_(2)
        tokenizer.missing_state.weight.fill_(3)
        tokenizer.masked_state.weight.fill_(5)
    tokens = tokenizer(batch)
    torch.testing.assert_close(tokens[:, 0, 0], torch.tensor([3., 5.]))
    missing = batch.missing.clone()
    missing[:, 0] = True
    values = batch.numeric_values.clone()
    values[:, 0] = float("nan")
    torch.testing.assert_close(tokenizer(replace(batch, numeric_values=values, missing=missing))[:, 0],
                               torch.full((2, 64), 4.))


def test_observed_unknown_missing_masked_and_padding_are_distinct(specs):
    tokenizer = FeatureTokenizer(specs)
    # Five copies of one record: observed category, unknown, real missing,
    # artificial mask, padding. Value placeholders in suppressed cells are ignored.
    flags = torch.zeros((5, 1), dtype=torch.bool)
    missing, masked, padding = flags.clone(), flags.clone(), flags.clone()
    missing[2] = True
    masked[3] = True
    padding[4] = True
    batch = FeatureBatch(torch.tensor([1]), torch.full((5, 1), float("nan")),
                         torch.tensor([[0], [2], [-123], [-456], [-789]]), missing, masked, padding)
    tokens = tokenizer(batch).squeeze(1)
    for i in range(5):
        for j in range(i):
            assert not torch.allclose(tokens[i], tokens[j])
    assert torch.count_nonzero(tokens[4]) == 0
    altered = replace(batch, categorical_values=torch.tensor([[0], [2], [500], [600], [700]]))
    torch.testing.assert_close(tokens, tokenizer(altered).squeeze(1))


def test_numeric_mask_never_reads_hidden_values(specs, batch):
    encoder = FeatureEncoder(specs, dropout=0).eval()
    masked = batch.masked.clone()
    masked[:, 0] = True
    hidden = replace(batch, masked=masked)
    values = batch.numeric_values.clone()
    values[:, 0] = float("nan")
    h, z = encoder(hidden)
    h2, z2 = encoder(replace(hidden, numeric_values=values))
    torch.testing.assert_close(h, h2)
    torch.testing.assert_close(z, z2)
    missing = batch.missing.clone()
    missing[:, 0] = True
    assert not torch.allclose(z, encoder(replace(batch, missing=missing))[1])
    z.square().sum().backward()
    assert encoder.tokenizer.masked_state.weight.grad.abs().sum() > 0


def test_feature_permutation_preserves_pooled_predictions_and_permuted_states(specs, batch):
    encoder = FeatureEncoder(specs).eval()
    head = nn.Linear(64, 2).eval()
    h, z = encoder(batch)
    order = torch.tensor([2, 0, 1])
    other_h, other_z = encoder(permute(batch, order))
    torch.testing.assert_close(other_h, h[:, order], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(head(other_z), head(z), rtol=1e-5, atol=2e-6)
    # Values must travel with identities; reassignment is a different record.
    changed = replace(batch, numeric_values=batch.numeric_values[:, [2, 1, 0]])
    assert not torch.allclose(encoder(changed)[1], z)


def test_padding_does_not_change_cls_and_all_padding_is_finite(specs, batch):
    encoder = FeatureEncoder(specs).eval()
    _, expected = encoder(batch)
    padded = FeatureBatch(torch.cat((batch.feature_ids, torch.tensor([-1]))),
                         torch.cat((batch.numeric_values, torch.full((2, 1), float("nan"))), 1),
                         torch.cat((batch.categorical_values, torch.full((2, 1), -99)), 1),
                         torch.cat((batch.missing, torch.zeros((2, 1), dtype=torch.bool)), 1),
                         torch.cat((batch.masked, torch.zeros((2, 1), dtype=torch.bool)), 1),
                         torch.cat((batch.padding, torch.ones((2, 1), dtype=torch.bool)), 1))
    h, z = encoder(padded)
    torch.testing.assert_close(z, expected, rtol=1e-5, atol=2e-6)
    assert h[:, -1].count_nonzero() == 0
    h, z = encoder(replace(batch, padding=torch.ones_like(batch.padding)))
    assert h.count_nonzero() == 0 and torch.isfinite(z).all()


def test_encoder_shape_architecture_and_train_eval_dropout(specs, batch):
    encoder = FeatureEncoder(specs, dropout=0.5)
    assert len(encoder.blocks) == 3
    for block in encoder.blocks:
        assert block.norm_first and block.self_attn.num_heads == 4
        assert block.linear1.in_features == 64 and block.linear1.out_features == 128
        assert block.self_attn.dropout == 0.5
        torch.testing.assert_close(block.activation(torch.tensor([-1.])), nn.functional.gelu(torch.tensor([-1.])))
    assert sum(p.numel() for p in encoder.parameters()) < 1_000_000
    first = encoder(batch)[1]
    assert not torch.equal(first, encoder(batch)[1])
    encoder.eval()
    h, z = encoder(batch)
    assert h.shape == (2, 3, 64) and z.shape == (2, 64)
    torch.testing.assert_close(z, encoder(batch)[1], rtol=0, atol=0)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/models/backbone.yaml").read_text())
    assert config["feature_encoder"]["width"] == encoder.width
    assert config["county_context"]["seeds"] == 4


def test_invalid_feature_ids_and_observed_values_fail(specs, batch):
    tokenizer = FeatureTokenizer(specs)
    for ids in (torch.tensor([0, 0, 2]), torch.tensor([0, 1, 7]), torch.tensor([-1, 1, 2])):
        with pytest.raises(ContractError):
            tokenizer(replace(batch, feature_ids=ids))
    values = batch.numeric_values.clone()
    values[0, 0] = float("nan")
    with pytest.raises(ContractError, match="nonfinite"):
        tokenizer(replace(batch, numeric_values=values))
    with pytest.raises(ContractError, match="positive"):
        FeatureSpec(name="x", kind="numeric", scale=0)
