"""Strict nested fitting with immutable, portable execution slices.

Reviewed endpoint adapters supply PreparedEndpoint. Geography audits splits but
never enters a predictor view. A passing slice may be checkpointed; only the
explicit complete flag permits consumption as a finished nuisance procedure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import re
import shutil
import tarfile
import time

import pandas as pd
import torch
import yaml

from oxyformer.contracts import CovariateView, OOFNuisances, SplitManifest, StageRequest, StageResult
from oxyformer.data.adapters.usaleep import primary_outcome_flags
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import LoadedData, validate_split
from oxyformer.design.eligibility import GeographyTable
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy, paired_records
from oxyformer.design.splits import InnerSplit, dependence_groups, tract_count
from oxyformer.execution.paths import atomic_json, output_path
from oxyformer.models.ablations import VARIANTS, build_variant
from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.likelihoods import CountyOffsets
from oxyformer.models.outcome import OutcomeTransformer
from oxyformer.models.tokens import FeatureSpec
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import (ArtifactRecord, ContractError, Immutable,
    canonical_json, file_hash, require, write_artifact)
from oxyformer.training import fit as fitting
from oxyformer.training.calibration import (AffineCalibration, CalibrationPartition, fit_affine,
    paired_tensors, pair_metrics, transfer_diagnostics)
from oxyformer.training.checkpoint import (CheckpointArtifact, CheckpointRequest, capture_rng,
    restore_rng, load_checkpoint, save_checkpoint, model_state_hash)
from oxyformer.training.fit import (FitConfig, FoldArtifacts, NuisanceSettings,
    subset, _inputs, _values, _partition, _weight_unit, _outcome_loss, _origin_loss,
    _backward_and_clip, _pooled_metrics, _bundle, _lineage, _tensor_state)
from oxyformer.training.pretrain import (StatefulSampler, SSLSettings, PretrainConfig, pretrain, environment_identity,
    fit_preprocessing)

SEEDS = (1103, 2207, 3301)
STAGES = ("primary", "ablation", "anchor", "refit-audit")
TRANSFORMER_VARIANTS = ("A0", "A1", "A2", "A3", "A4")


def digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedEndpoint(Immutable):
    """Adapter output: reviewed data, frozen target, splits and model routing.

    No raw source-column interpretation is performed by the controller. Source
    adapters and the design stage remain authoritative for these identities.
    """
    data: LoadedData
    entity_graph: EntityGraph
    geography: GeographyTable
    outer: SplitManifest
    inner: tuple[InnerSplit, ...]
    policy: ShiftOrStayPolicy
    policy_covariates: PolicyCovariates
    treatment_design: TreatmentDesign
    feature_kinds: tuple[tuple[str, str], ...]
    families: tuple[tuple[str, ...], ...]
    county_field: str
    exposure_assignment_level: str
    family: str = "identity"
    population_field: str | None = None

    def validate(self, fold):
        require(len({i.outer_fold for i in self.inner}) == len(self.inner), "duplicate inner parent")
        inners = {i.outer_fold: i for i in self.inner}
        require(fold in inners, "missing inner split")
        inner = inners[fold]
        _validate_geography(self.data, self.entity_graph, self.outer, inner,
                            self.geography, self.county_field, fold)
        return inner

    def configuration(self, fold, output_dir, **kwargs):
        config = FitConfig(data=self.data, entity_graph=self.entity_graph, inner=self.validate(fold), fold=fold,
            policy=self.policy, policy_covariates=self.policy_covariates,
            treatment_design=self.treatment_design, feature_kinds=self.feature_kinds,
            families=self.families, county_field=self.county_field,
            exposure_assignment_level=self.exposure_assignment_level, family=self.family,
            population_field=self.population_field, output_dir=str(output_dir), **kwargs)
        _check_fitting_minima(config, self.outer, self.geography)
        return config


def _whole_groups(groups, assignments):
    for group in groups:
        present = set(group).intersection(assignments)
        if present:
            require(present == set(group) and len({assignments[i] for i in group}) == 1,
                    "geographic component crosses fitting partitions")


def _split_assignments(split):
    assignments = dict(zip(split.original_ids, split.fold_ids))
    assignments.update((oid, "design") for oid in split.design_ids)
    assignments.update((oid, "excluded") for oid in split.excluded_ids)
    return assignments


def _validate_geography(data, graph, outer, inner, geography, county_field, fold, stopping_ids=()):
    """One geographic contract for adapters, direct calls, and stopping.

    Use the design producer's transitive closure over entity links, tract,
    assignment geography and subblock. EntityGraph alone is not that closure.
    """
    manifest = data.manifest
    validate_split(outer, manifest, graph)
    require(outer.level == "outer" and set(outer.fold_ids) == set(range(5)),
            "five outer geographic folds required")
    require(outer.seed_ids == SEEDS, "registered seeds required")
    require(type(fold) is int and fold in range(5), "invalid outer fold")
    require(inner.outer_fold == fold, "wrong inner parent fold")
    validate_split(inner.split, inner.data_manifest, inner.entity_graph)
    manifest.spec.assert_compatible(inner.split.spec)
    require(inner.split.level == "inner" and inner.split.seed_ids == SEEDS and
            set(inner.split.fold_ids) == {0, 1, 2}, "three inner geographic folds required")
    training = outer.training_ids(fold)
    require(set(inner.data_manifest.original_ids) == set(training), "inner escapes outer training")
    for field in ("schema", "registry", "sources", "id_field", "outcome_field", "exposure_field", "weight_field"):
        require(getattr(inner.data_manifest, field) == getattr(manifest, field), "inner data identity mismatch")
    require(type(geography) is GeographyTable and geography.data_manifest_hash == manifest.content_hash,
            "geography must bind the fitting manifest")
    rows = {row.original_id: row for row in geography.rows}
    require(set(rows) == set(manifest.original_ids), "geographic coverage mismatch")
    routes = dict(zip(manifest.original_ids, data.county_routing(county_field)))
    require(all(rows[i].county == routes[i] for i in rows), "county route mismatch")
    flags = primary_outcome_flags()
    require(all(rows[i].outcome_flag in flags and rows[i].label_available for i in outer.original_ids),
            "non-primary or unavailable labels forbidden")
    groups = dependence_groups(geography.rows, graph)
    _whole_groups(groups, _split_assignments(outer))
    _whole_groups(groups, _split_assignments(inner.split))
    counties = {rows[i].county for i in outer.original_ids}
    for county in counties:
        require({f for i, f in zip(outer.original_ids, outer.fold_ids) if rows[i].county == county}
                == set(range(5)), "outer folds must be within counties")
        require(tract_count([rows[i] for i in training], county) >= 8,
                "fewer than eight outer-training tracts")
        require({f for i, f in zip(inner.split.original_ids, inner.split.fold_ids) if rows[i].county == county}
                == {0, 1, 2}, "inner folds must be within counties")
    stops = dict(stopping_ids)
    require(len(stops) == len(stopping_ids) and set(stops) <= {0, 1, 2}, "invalid stopping folds")
    for inner_fold in range(3):
        allowed = set(inner.split.training_ids(inner_fold))
        stopping = tuple(stops.get(inner_fold, ()))
        require(len(set(stopping)) == len(stopping) and set(stopping) < allowed,
                "stopping records must be inside the fitting partition")
        ids = allowed - set(stopping)
        assignments = _split_assignments(inner.split)
        assignments.update((i, "fit" if i in ids else "stop") for i in allowed)
        _whole_groups(groups, assignments)
        for county in counties:
            require(tract_count([rows[i] for i in ids], county) >= 4,
                    "fewer than four inner-fitting tracts after stopping reservation")
    return groups


def _check_fitting_minima(config, outer, geography):
    return _validate_geography(config.data, config.entity_graph, outer, config.inner,
                               geography, config.county_field, config.fold, config.stopping_ids)


def _build(bundle, *, encoder_state=None):
    variant = bundle.get("variant", "A0")
    if variant == "A0":
        return fitting._build(bundle, encoder_state=encoder_state)
    features = tuple(FeatureSpec.from_json(v) for v in bundle["preprocessing"])
    encoder = FeatureEncoder(features, dropout=bundle["dropout"]).float()
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    split = SplitManifest.from_json(bundle["split"])
    references = CovariateView.from_json(bundle["references"])
    context = None if variant == "A2" else CountyContext(encoder, references, split, 0,
        tuple(bundle["counties"]), county_field=bundle["county_field"],
        checkpoint_hash=bundle["ssl_hash"], dropout=bundle["dropout"])
    def offsets(family):
        return CountyOffsets(split, 0, tuple(bundle["counties"]), family=family,
            exposure_assignment_level=bundle["exposure_assignment_level"])
    pair = build_variant(variant, encoder,
        treatment_design=TreatmentDesign.from_json(bundle["treatment_design"]),
        raw_x_dim=sum((1 if f.kind == "numeric" else len(f.categories) + 1) + 1 for f in features),
        family=bundle["family"], county_context=context, outcome_offsets=offsets(bundle["family"]),
        origin_offsets=offsets("bernoulli"), dropout=bundle["dropout"])
    model = pair.outcome if bundle["kind"] == "outcome" else pair.correction
    if "state" in bundle:
        model.load_state_dict(bundle["state"], strict=True)
    return model.float()


def _predict(model, view, inputs, policy, *, outcome_mean=False, base_only=False):
    require(type(view) is CovariateView and view.use == "nuisance", "label-free nuisance view required")
    require(view.original_ids == inputs.original_ids, "prediction alignment mismatch")
    require(set(inputs.counties) <= set(model.group_offsets.counties), "unseen county prediction requires a transfer experiment")
    action = policy.apply(inputs.a_mmhg, inputs.policy_covariates)
    query = torch.tensor(tuple(zip(action.a_mmhg, action.d_mmhg)), dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(-1)
    batch = model.encoder.tokenizer.prepare(view)
    raw = fitting._raw(batch, model.encoder.tokenizer.features)
    context = (query.new_zeros((len(view.original_ids), 4, 64)) if model.county_context is None
               else model.county_context(inputs.original_ids, inputs.counties))
    offset = query.new_zeros(len(view.original_ids)) if base_only else model.group_offsets(inputs.counties)
    method = ((model.mean if outcome_mean else model.linear_predictor)
              if isinstance(model, OutcomeTransformer) else model.logits)
    result = method(query, batch, raw, context, offset)
    require(bool(torch.isfinite(result).all()), "nonfinite nuisance predictions")
    return result


def _profile(model, config, view, inputs):
    if isinstance(model, OutcomeTransformer) and config.family == "identity":
        model.eval()
        with torch.no_grad():
            base = _predict(model, view, inputs, config.policy, base_only=True)[:, 0]
            target = torch.tensor(_values(config, config.data.manifest.outcome_field, view.original_ids),
                                  dtype=torch.float32, device=base.device)
            units = {c: _weight_unit([w for county, w in zip(inputs.counties, inputs.origin_weights)
                                     if county == c]) for c in set(inputs.counties)}
            weights = torch.tensor([w / units[c] for c, w in zip(inputs.counties, inputs.origin_weights)],
                                   dtype=torch.float64, device=base.device)
            model.group_offsets.update_identity(view.original_ids, target.double(), base.double(), weights)


def _scratch(config, split, ids, view, root, seed):
    """A1 fits preprocessing only; it does not perform or count an SSL fit."""
    view = subset(view, ids, use="ssl")
    settings = SSLSettings(fold=0, feature_kinds=config.feature_kinds, families=config.families)
    features = fit_preprocessing(view, settings)
    torch.manual_seed(seed)
    encoder = FeatureEncoder(features).float()
    tensors = {"encoder." + k: v for k, v in encoder.state_dict().items()}
    identity = fitting.CheckpointIdentity(training_ids=ids, data_hash=view.content_hash,
        split_hash=split.content_hash, config_hash=digest([settings.to_dict(), "no-ssl"]),
        preprocessing_hash=digest([f.to_dict() for f in features]), seed=seed,
        scientific_code_hash=digest("preprocessing-only"), environment=environment_identity(torch.device(config.device)))
    lineage = _lineage(config.data.manifest, identity, ids, (view.content_hash,),
        model_hash=model_state_hash(tensors), parameter_count=sum(v.numel() for v in tensors.values()))
    root.mkdir(parents=True)
    return save_checkpoint(root, identity=identity, lineage=lineage,
        state={"model": tensors, "best_model": tensors, "preprocessing": tuple(f.to_json() for f in features),
               "progress": {"epoch": 0, "step": 0}, "initialization_kind": "preprocessing_only"},
        complete=True, reason="max_epochs")


def _train_transaction(config, bundle, encoder_state, view, stopping, epochs, seed, budget, saved=None):
    """One transaction per original-record minibatch, including both copies."""
    torch.manual_seed(seed)
    model = _build(bundle, encoder_state=encoder_state).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=bundle["learning_rate"],
                                 weight_decay=config.settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)
    inputs = _inputs(config, view.original_ids)
    require(sum(inputs.origin_weights) > 0, "fitting partition has no target mass")
    weight_unit = _weight_unit(inputs.origin_weights)
    original_mass = sum(w / weight_unit for w in inputs.origin_weights)
    sampler = StatefulSampler(len(view.original_ids), seed)
    progress = dict(epoch=0, step=0, phase="start", best_loss=None, best_epoch=0, bad_epochs=0, history=[])
    best = None
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        progress, best = saved["progress"], saved["best"]
        restore_rng(saved["rng"], config.device)
    complete, reason = False, None
    while not complete:
        reason = budget.reason()
        if reason:
            break
        if progress["phase"] == "start":
            _profile(model, config, view, inputs)
            progress["phase"] = "train"
        if progress["phase"] == "train":
            indices = sampler.indices(config.settings.batch_size)
            ids = tuple(view.original_ids[i] for i in indices)
            part = subset(view, ids)
            metadata = _inputs(config, ids)
            if sum(metadata.origin_weights) == 0:
                # A sampled zero-mass group has no objective and no decay update.
                sampler.cursor += len(indices)
                progress["step"] += 1
                budget.batches += 1
                if sampler.cursor == sampler.size:
                    progress["phase"] = "evaluate"
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted = _predict(model, part, metadata, config.policy)
            if bundle["kind"] == "outcome":
                loss = _outcome_loss(config, predicted[:, 0], ids, weight_unit=weight_unit)
                mass = original_mass
            else:
                pairs = paired_records(config.policy.apply(metadata.a_mmhg, metadata.policy_covariates),
                                       metadata.origin_weights, weight_id=config.data.manifest.spec.weight_id)
                # The authoritative records retain raw origin weights. Only
                # the numerical loss receives weights in common arithmetic
                # units, using the SAME unit as the full fitting denominator.
                loss = _origin_loss(predicted.double(), pairs, weight_unit)
                mass = 2 * original_mass
            # Uniform sampling of ORIGINALS estimates the global weighted loss.
            # Dividing each batch by its own mass would optimize a different law.
            loss = loss * (len(view.original_ids) / (len(ids) * mass))
            require(bool(torch.isfinite(loss)), "nonfinite nuisance loss")
            _backward_and_clip(loss, model.parameters(), config.settings.gradient_norm)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                    "nonfinite nuisance parameters")
            sampler.cursor += len(indices)
            progress["step"] += 1
            budget.batches += 1
            if sampler.cursor == sampler.size:
                progress["phase"] = "evaluate"
            continue
        _profile(model, config, view, inputs)
        model.eval()
        score = None
        # A zero-mass stopping set has no checkpoint-selection information;
        # retain the final epoch of the already-declared training bound.
        if stopping and sum(_inputs(config, stopping).origin_weights) > 0:
            stop_view = subset(config.data.covariates(view.columns), stopping)
            stop_inputs = _inputs(config, stopping)
            with torch.no_grad():
                predicted = _predict(model, stop_view, stop_inputs, config.policy)
                score = (float(_outcome_loss(config, predicted[:, 0], stopping, reduction="mean"))
                         if bundle["kind"] == "outcome" else pair_metrics(predicted, stop_inputs.origin_weights)[0])
            require(math.isfinite(score), "nonfinite stopping loss")
        progress["epoch"] += 1
        progress["history"].append(score)
        if score is None or progress["best_loss"] is None or score < progress["best_loss"]:
            progress.update(best_loss=score, best_epoch=progress["epoch"], bad_epochs=0)
            best = deepcopy(model.state_dict())
        else:
            progress["bad_epochs"] += 1
        scheduler.step()
        complete = progress["epoch"] >= epochs or progress["bad_epochs"] >= config.settings.patience
        if complete:
            reason = "max_epochs" if progress["epoch"] >= epochs else "patience"
        else:
            sampler.finish_epoch()
            progress["phase"] = "start"
    state = dict(model=deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), rng=capture_rng(config.device),
        progress=progress, best=best)
    if complete:
        model.load_state_dict(best, strict=True)
        model.eval()
    return model, state, complete, reason


# Merged SSL numerical/checkpoint algorithm with design-owned stopping groups.
def _grouped_ssl(config, split, fitting, view, root, seed, budget, predecessor, geographic_groups):
    groups = [g for g in geographic_groups if set(g) <= set(fitting)]
    require(len(groups) >= 2, "SSL needs two independent fitting components")
    # A deterministic component draw, shared by all later transformed copies.
    stop = min(groups, key=lambda group: digest([seed, group, "ssl-stopping"]))
    settings = SSLSettings(fold=0, feature_kinds=config.feature_kinds, families=config.families,
                          max_epochs=config.ssl_epochs, stopping_ids=stop)
    remaining = (None if config.max_batches is None else config.max_batches - budget.batches)
    seconds = max(.01, config.slice_seconds - (time.monotonic() - budget.started))
    ssl_config = PretrainConfig(settings=settings, output_dir=str(root), device=config.device,
        predecessor=predecessor, stop_request=budget.request, max_batches=remaining,
        slice_seconds=seconds, checkpoint_margin_seconds=min(config.checkpoint_margin_seconds, seconds / 2))
    artifact = pretrain(subset(view, fitting, use="ssl"), split, ssl_config, seed)
    # SSL validation transactions are also charged to the enclosing slice.
    state = load_checkpoint(artifact, artifact.identity)
    before = load_checkpoint(predecessor, predecessor.identity) if predecessor else None
    def transactions(saved):
        if saved is None:
            return 0
        progress = saved["progress"]
        size = len(saved["stopping_ids"])
        count = progress["step"] + progress["epoch"] * math.ceil(size / settings.batch_size)
        if progress["phase"] == "validation" and progress["validation_cursor"] < size:
            count += math.ceil(progress["validation_cursor"] / settings.batch_size)
        return count
    budget.batches += transactions(state) - transactions(before)
    return artifact


fitting_module_ssl = _grouped_ssl

def _advance(config, outer, manifest, identity, controller, root, budget, variant, groups):
    columns = tuple(name for name, _ in config.feature_kinds)
    all_view = config.data.covariates(columns)
    outer_ids = outer.training_ids(config.fold)
    tasks = []
    partitions = {}
    audits = {}
    for fold in range(3):
        stopping = tuple(dict(config.stopping_ids).get(fold, ()))
        local, fitting, held = _partition(config, config.inner.split, fold, stopping=stopping)
        audit = CalibrationPartition(fold=fold, evaluation_ids=held, fitting_ids=fitting,
            checkpoint_ids=stopping, frozen_epochs=None if stopping else config.settings.frozen_epochs)
        audit.validate(outer_ids)
        partitions[fold] = (local, fitting, held, stopping)
        audits[fold] = audit
        tasks.append(("ssl", fold, None))
        tasks.extend((kind, fold, grid) for kind in ("outcome", "origin") for grid in range(4))
    outer_local, fitting, held = _partition(config, outer, config.fold)
    partitions[3] = (outer_local, fitting, held, ())
    tasks.extend((("select", 3, None), ("ssl", 3, None), ("outcome", 3, None), ("origin", 3, None)))
    while controller["position"] < len(tasks):
        reason = budget.reason()
        if reason:
            return controller, False, reason
        kind, fold, grid = tasks[controller["position"]]
        local, fitting, held, stopping = partitions[fold]
        if kind == "select":
            selected = {}
            for model_kind in ("outcome", "origin"):
                candidates = []
                for g in range(4):
                    values = [r for r in controller["results"] if r["kind"] == model_kind and r["grid"] == g]
                    scores = _pooled_metrics(values)
                    candidates.append((scores, g))
                chosen = min(candidates)[1]
                values = [r for r in controller["results"] if r["kind"] == model_kind and r["grid"] == chosen]
                selected[model_kind] = dict(grid=chosen, epochs=int(statistics.median(r["epochs"] for r in values)))
            # Each OOF classifier is chosen without consulting its own
            # evaluation scores, including indirectly through other folds.
            origin = [min((r for r in controller["results"]
                           if r["kind"] == "origin" and r["fold"] == fold),
                          key=lambda r: (r["selection_metrics"], r["grid"]))
                      for fold in range(3)]
            selected["origin"]["calibration_grids"] = tuple(r["grid"] for r in origin)
            ids = tuple(oid for r in origin for oid in r["ids"])
            logits = [z for r in origin for z in r["predictions"]]
            weights = _inputs(config, ids).origin_weights
            calibration = fit_affine(logits, weights, original_ids=ids,
                fold_ids=tuple(r["fold"] for r in origin for _ in r["ids"]),
                partitions=tuple(audits.values()), outer_training_ids=outer_ids,
                lineage=_lineage(manifest, identity, ids, (manifest.content_hash, config.inner.content_hash)))
            controller.update(selection=selected, calibration=calibration.to_json(), oof_logits=logits)
        elif kind == "ssl":
            pending = (CheckpointArtifact.from_json(controller["ssl_pending"])
                       if controller["ssl_pending"] else None)
            if pending is None and variant != "A1":
                controller["counts"]["ssl_fits"] += 1
            artifact = (_scratch(config, local, fitting, all_view, root / f"preprocess-{fold}", identity.seed)
                        if variant == "A1" else fitting_module_ssl(config, local, fitting, all_view,
                            root / f"ssl-{fold}", identity.seed, budget, pending, groups))
            if not artifact.complete:
                controller["ssl_pending"] = artifact.to_json()
                return controller, False, artifact.reason
            controller["initializations"][fold] = artifact.to_json()
            controller["ssl_pending"] = None
        else:
            if fold == 3:
                grid = controller["selection"][kind]["grid"]
                epochs = controller["selection"][kind]["epochs"]
            else:
                epochs = config.settings.max_epochs if stopping else config.settings.frozen_epochs
            initialization = CheckpointArtifact.from_json(controller["initializations"][fold])
            bundle, encoder_state = _bundle(config, local, fitting, initialization, kind, config.settings.grid[grid])
            bundle["variant"] = variant
            view = subset(all_view, fitting)
            if controller["active"] is None:
                controller["counts"]["nuisance_fits"] += 1
            model, state, complete, reason = _train_transaction(config, bundle, encoder_state, view, stopping,
                epochs, identity.seed, budget, controller["active"])
            controller["parameter_count"] = sum(p.numel() for p in model.parameters())
            if not complete:
                controller["active"] = state
                return controller, False, reason
            controller["active"] = None
            if fold == 3:
                with torch.no_grad():
                    _predict(model, view, _inputs(config, fitting), config.policy,
                             outcome_mean=kind == "outcome")
                bundle["state"] = deepcopy(model.state_dict())
                controller["final"][kind] = bundle
            else:
                metadata = _inputs(config, held)
                with torch.no_grad():
                    predicted = _predict(model, subset(all_view, held), metadata, config.policy)
                    metrics = ((float(_outcome_loss(config, predicted[:, 0], held, reduction="mean")),)
                               if kind == "outcome" else pair_metrics(predicted, metadata.origin_weights))
                require(all(math.isfinite(m) for m in metrics), "nonfinite held-out factual loss")
                mass_unit = _weight_unit(metadata.origin_weights)
                exact_mass = sum(map(Fraction, metadata.origin_weights), Fraction())
                result = dict(kind=kind, fold=fold, grid=grid, ids=held,
                    predictions=predicted.tolist(), metrics=metrics,
                    mass=sum(w / mass_unit for w in metadata.origin_weights), mass_unit=mass_unit,
                    mass_ratio=(exact_mass.numerator, exact_mass.denominator),
                    epochs=state["progress"]["best_epoch"], ownership=audits[fold].to_json())
                if kind == "origin":
                    # With frozen epochs, an in-sample fitting score may choose
                    # a grid, but never a calibration evaluation score. Explicit
                    # stopping partitions provide an independent tuning score.
                    selection_ids = stopping or fitting
                    require(set(selection_ids).isdisjoint(held),
                            "calibration records entered grid selection")
                    selection_inputs = _inputs(config, selection_ids)
                    with torch.no_grad():
                        selection_logits = _predict(model, subset(all_view, selection_ids),
                                                    selection_inputs, config.policy)
                    result.update(selection_ids=selection_ids,
                        selection_metrics=pair_metrics(selection_logits, selection_inputs.origin_weights))
                controller["results"].append(result)
        controller["position"] += 1
    calibration = AffineCalibration.from_json(controller["calibration"])
    model = _build(controller["final"]["origin"]).to(config.device).eval()
    metadata = _inputs(config, calibration.original_ids)
    with torch.no_grad():
        refit = _predict(model, subset(all_view, calibration.original_ids), metadata, config.policy)
    positive_refit, _, _ = paired_tensors(refit, metadata.origin_weights)
    calibration.ratios(positive_refit)
    diagnostics = transfer_diagnostics(calibration, controller["oof_logits"], refit.cpu(),
        metadata.origin_weights, lineage=_lineage(manifest, identity, calibration.original_ids,
            (calibration.content_hash, model_state_hash(_tensor_state(controller)))))
    controller["transfer_diagnostics"] = diagnostics.to_json()
    controller["parameter_count"] = sum(sum(p.numel() for p in _build(bundle).parameters())
                                          for bundle in controller["final"].values())
    write_artifact(root / "calibration-transfer.json", diagnostics)
    write_artifact(root / "calibration.json", calibration)
    return controller, True, "max_epochs"


def run_fold(config, outer, seed, *, geography, variant="A0"):
    """One resumable endpoint/outer-fold/seed; no mutable cache survives a call."""
    require(variant in TRANSFORMER_VARIANTS, "variant needs a compatible resumable nuisance adapter")
    manifest = config.data.manifest
    allowed = fitting._validate(manifest.spec, outer, manifest, config, seed)
    groups = _check_fitting_minima(config, outer, geography)
    destination = Path(config.output_dir)
    require(not any(p.is_symlink() for p in (destination, *destination.parents)),
            "symlink in fitting output path")
    config = replace(config, device=str(fitting.resolve_device(config.device)))
    with fitting._numerics(config.device):
        identity = fitting._science_identity(config, outer, manifest, seed, allowed)
        identity = replace(identity, config_hash=digest([identity.config_hash, variant, geography.content_hash]))
        controller = dict(position=0, initializations={}, ssl_pending=None, active=None,
            results=[], final={}, selection=None, calibration=None,
            counts={"ssl_fits": 0, "nuisance_fits": 0, "batches": 0},
            compute={"wall_seconds": 0., "cpu_seconds": 0., "gpu_seconds": 0., "device": config.device,
                     "gpu_seconds_scope": "single_device_elapsed_not_kernel_utilization"})
        if config.predecessor is not None:
            require(not config.predecessor.complete, "fold is already complete")
            require(config.predecessor.fold == config.fold and config.predecessor.seed == seed,
                    "continuation fold/seed mismatch")
            controller = load_checkpoint(config.predecessor.checkpoint, identity)["controller"]
        root = Path(config.output_dir).resolve() / "nuisance"
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700)
        request = config.stop_request or CheckpointRequest()
        budget = fitting._Budget(config, request)
        cpu_started = time.process_time()
        with request.signals():
            controller, complete, reason = _advance(config, outer, manifest, identity,
                controller, root, budget, variant, groups)
        controller["counts"]["batches"] += budget.batches
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(config.device)
        elapsed = time.monotonic() - budget.started
        controller["compute"]["wall_seconds"] += elapsed
        if torch.device(config.device).type == "cuda":
            controller["compute"]["gpu_seconds"] += elapsed
        controller["compute"]["cpu_seconds"] += time.process_time() - cpu_started
        tensors = fitting._tensor_state(controller)
        lineage = _lineage(manifest, identity, allowed, (manifest.content_hash, outer.content_hash),
            model_hash=model_state_hash(tensors), parameter_count=controller.get("parameter_count", 0))
        checkpoint = save_checkpoint(root, identity=identity, lineage=lineage,
            state={"model": tensors, "controller": controller,
                   "progress": {"epoch": controller["position"], "step": controller["position"]}},
            complete=complete, reason=reason,
            predecessor=config.predecessor.checkpoint if config.predecessor else None)
        held = tuple(i for i, f in zip(outer.original_ids, outer.fold_ids) if f == config.fold)
        return FoldArtifacts(spec=manifest.spec, split=outer, data_manifest=manifest,
            fold=config.fold, seed=seed, prediction_inputs=_inputs(config, held), checkpoint=checkpoint)


def predict(artifacts, view, policy, *, device=None):
    """Frozen prediction accepts only copied X and routing; never LoadedData."""
    require(type(artifacts) is FoldArtifacts and artifacts.complete, "unfinished procedure cannot predict")
    require(type(view) is CovariateView and view.use == "nuisance", "label-free nuisance view required")
    artifacts.spec.assert_compatible(view.spec)
    require(policy.policy_id == artifacts.spec.policy_id, "prediction policy mismatch")
    require(set(view.original_ids) == set(artifacts.prediction_inputs.original_ids), "held-out IDs mismatch")
    inputs = fitting._take_inputs(artifacts.prediction_inputs, view.original_ids)
    controller = load_checkpoint(artifacts.checkpoint, artifacts.checkpoint.identity)["controller"]
    selected = fitting.resolve_device(device or dict(artifacts.checkpoint.identity.environment)["device"])
    with fitting._numerics(selected), torch.no_grad():
        mu = _predict(_build(controller["final"]["outcome"]).to(selected).eval(), view, inputs, policy, outcome_mean=True)
        calibration = AffineCalibration.from_json(controller["calibration"])
        prediction_environment = environment_identity(selected)
        ratio = (torch.ones_like(mu) if policy.is_identity else calibration.ratios(
            _predict(_build(controller["final"]["origin"]).to(selected).eval(), view, inputs, policy)))
    lineage = replace(artifacts.checkpoint.lineage, unit_ids=view.original_ids, environment=prediction_environment,
        parent_hashes=(artifacts.data_manifest.content_hash, artifacts.checkpoint.content_hash, view.content_hash))
    return OOFNuisances(spec=artifacts.spec, original_ids=view.original_ids,
        fold_ids=(artifacts.fold,) * len(view.original_ids), seed_ids=(artifacts.seed,) * len(view.original_ids),
        mu_a=tuple(mu[:, 0].double().tolist()), mu_d=tuple(mu[:, 1].double().tolist()),
        r_a=tuple(ratio[:, 0].double().tolist()), r_d=tuple(ratio[:, 1].double().tolist()),
        origin_weights=inputs.origin_weights, lineage=lineage)


def _components(controller):
    values = list(controller["initializations"].values())
    if controller["ssl_pending"] is not None:
        values.append(controller["ssl_pending"])
    return tuple(CheckpointArtifact.from_json(value) for value in values)


def export_continuation(artifact, path, *, binding, task_id, chain, allowed_root):
    """Bundle all live components, not pointers to previous attempt directories."""
    controller = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]
    blobs = {}
    for component in (artifact.checkpoint,) + _components(controller):
        location = Path(component.path).resolve(strict=True)
        require(location.is_relative_to(Path(allowed_root).resolve()), "checkpoint outside current attempt")
        load_checkpoint(component, component.identity)
        blobs["checkpoints/" + component.sha256 + ".ofc"] = location.read_bytes()
    header = {"version": 1, "fold": artifact.to_dict(), "binding": binding,
              "task_id": task_id, "chain": chain}
    blobs["bundle.json"] = canonical_json(header).encode()
    # The destination is private and create-once; StageResult publishes its digest
    # only after close. A crash cannot expose a passing partial archive.
    with Path(path).open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in sorted(blobs.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o400, 0
            archive.addfile(info, BytesIO(payload))


def import_continuation(path, root, *, binding, chain):
    """Relocate by verified blob digest; serialized absolute paths are never read."""
    imported = Path(root) / "predecessor"
    imported.mkdir()
    # Explicit TAR decoding: a tar containing ZIP checkpoints can itself pass
    # zipfile.is_zipfile, so generic format sniffing is inappropriate here.
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), "duplicate continuation member")
        require(sum(m.size for m in members) <= 10 * 1024**3, "continuation exceeds byte limit")
        for member in members:
            require(member.isfile() and (member.name == "bundle.json" or
                re.fullmatch(r"checkpoints/[0-9a-f]{64}\.ofc", member.name) is not None),
                "unexpected continuation member or link")
        for member in members:
            target = imported / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    header = json.loads((imported / "bundle.json").read_text())
    require(header["version"] == 1 and header["binding"] == binding, "continuation fingerprint mismatch")
    previous_chain = header["chain"]
    require(header["task_id"] == chain["predecessor"] and previous_chain["owner"] == chain["owner"]
            and previous_chain["step"] + 1 == chain["step"], "undeclared or nonconsecutive predecessor")
    original = FoldArtifacts.from_json(canonical_json(header["fold"]))
    used = {"bundle.json"}
    def local(component):
        name = "checkpoints/" + component.sha256 + ".ofc"
        used.add(name)
        rebound = replace(component, path=str(imported / name))
        load_checkpoint(rebound, rebound.identity)
        return rebound
    checkpoint = local(original.checkpoint)
    state = load_checkpoint(checkpoint, checkpoint.identity)
    controller = state["controller"]
    controller["initializations"] = {k: local(CheckpointArtifact.from_json(v)).to_json()
                                     for k, v in controller["initializations"].items()}
    if controller["ssl_pending"] is not None:
        controller["ssl_pending"] = local(CheckpointArtifact.from_json(controller["ssl_pending"])).to_json()
    require({str(p.relative_to(imported)) for p in imported.rglob("*") if p.is_file()} == used,
            "unexpected continuation members")
    rebound_root = Path(root) / "relocated"
    rebound_root.mkdir()
    checkpoint = save_checkpoint(rebound_root, identity=checkpoint.identity, lineage=checkpoint.lineage,
        state=state, complete=checkpoint.complete, reason=checkpoint.reason)
    return replace(original, checkpoint=checkpoint)


def _dependency(request, config, reference):
    """Resolve exactly one declared dependency; never scan sibling attempts."""
    require(type(reference) is dict and set(reference) == {"dependency", "path"}, "invalid dependency reference")
    from oxyformer.provenance import relative_artifact_path
    relative_artifact_path(reference["path"])
    roots = config["dependencies"]
    require(reference["dependency"] in roots, "dependency is not declared")
    root = Path(roots[reference["dependency"]]).resolve(strict=True)
    path = (root / reference["path"]).resolve(strict=True)
    require(path.is_relative_to(root) and str(path) in request.dependency_paths, "undeclared dependency file")
    return path


def _settings(lock, endpoint):
    settings = lock.get("nested_cv", {})
    require(set(settings) <= {"ssl_epochs", "frozen_epochs", "batch_size", "stopping_ids", "precision", "synthetic"},
            "unknown nested recipe field")
    require(settings.get("precision", "fp32") == "fp32", "BF16 disabled pending a measured precision audit")
    ssl_epochs, epochs = settings.get("ssl_epochs", 30), settings.get("frozen_epochs", 150)
    if ssl_epochs != 30 or epochs != 150:
        require(settings.get("synthetic") is True and endpoint.data.manifest.spec.endpoint.startswith("synthetic"),
                "shortened training is only registered for synthetic fixtures")
    return dict(ssl_epochs=ssl_epochs,
        settings=NuisanceSettings(batch_size=settings.get("batch_size", 256), frozen_epochs=epochs),
        stopping_ids=tuple((int(f), tuple(ids)) for f, ids in settings.get("stopping_ids", [])))


def run_stage(request: StageRequest) -> StageResult:
    """Run primary/ablation/anchor/refit tasks from declared endpoint artifacts.

    Task parameters bind endpoint_input (dependency/path), endpoint, target_id,
    outer_fold, seed and variant. The existing runner supplies dependency roots,
    recipe_lock and the owner/step/predecessor chain. Training settings reside in
    recipe_lock.nested_cv; execution-only limits reside in task.slice.
    """
    try:
        request.verify_inputs()
        require(request.stage in STAGES, "unregistered nested stage")
        config = yaml.safe_load(Path(request.config_path).read_text())
        task = json.loads(Path(request.task_path).read_text())
        require(task["stage"] == request.stage, "stage mismatch")
        parameters = task["parameters"]
        variant = parameters.get("variant", "A0")
        require(variant in VARIANTS, "unknown variant")
        if variant not in TRANSFORMER_VARIANTS:
            return StageResult(request_hash=request.content_hash, status="blocked", artifacts=(),
                message="Registered variant requires a resumable adapter for signed corrections or foundation state; "
                        "the merged density-ratio checkpoint contract cannot represent it.")
        require(request.stage == "ablation" or variant == "A0", "variants require an ablation task")
        endpoint_path = _dependency(request, config, parameters["endpoint_input"])
        endpoint = PreparedEndpoint.from_json(endpoint_path.read_text())
        spec = endpoint.data.manifest.spec
        require(parameters["endpoint"] == spec.endpoint and parameters["target_id"] == spec.target_id,
                "endpoint/target mismatch")
        if request.stage == "refit-audit":
            deletion = parameters["deletion"]
            require(deletion["kind"] in ("county", "state") and deletion["parent_target"] != spec.target_id,
                    "deletion requires a separately frozen refit target")
            require(all(getattr(row, deletion["kind"]) != deletion["value"] for row in endpoint.geography.rows),
                    "deleted geography remains in refit data")
        lock_ref = task["recipe_lock"]
        lock_path = _dependency(request, config, {k: lock_ref[k] for k in ("dependency", "path")})
        require(file_hash(lock_path) == lock_ref["sha256"], "recipe hash mismatch")
        lock = json.loads(lock_path.read_text())
        chain = task["continuation"]
        require(set(chain) == {"owner", "step", "predecessor"} and bool(chain["owner"])
                and type(chain["step"]) is int and chain["step"] >= 0, "invalid continuation chain")
        require((chain["step"] == 0) == (chain["predecessor"] is None), "missing predecessor")
        destination = Path(request.output_dir)
        require(not any(p.is_symlink() for p in (destination, *destination.parents)),
                "symlink in stage output path")
        root = destination.resolve(strict=True)
        # Validate every write destination before importing or fitting anything.
        # In particular, pandas' parquet writer otherwise follows an existing link.
        for name in ("work", "predecessor", "relocated", "continuation.tar", "progress.json",
                     "artifact_manifest.json", "nuisances.parquet", "model_bundle.tar", "metrics.json"):
            require(not output_path(root, name).exists(), "attempt output already exists: " + name)
        binding = {"code": request.code_identity, "environment": list(map(list, fitting.fit_environment(parameters.get("device", "auto")))),
            "recipe": lock_ref["sha256"], "endpoint": endpoint.content_hash,
            "parameters": parameters, "stage": request.stage,
            "source_code": digest([(str(p.relative_to(Path(__file__).parents[1])), file_hash(p))
                for p in sorted(Path(__file__).parents[1].rglob("*.py"))])}
        predecessor = None
        if chain["predecessor"] is not None:
            archive = _dependency(request, config, {"dependency": chain["predecessor"], "path": "continuation.tar"})
            predecessor = import_continuation(archive, root, binding=binding, chain=chain)
        limits = task.get("slice", {})
        require(set(limits) <= {"seconds", "margin_seconds", "max_batches"}, "unknown slice limit")
        options = _settings(lock, endpoint)
        fit_config = endpoint.configuration(parameters["outer_fold"], root / "work",
            **options, device=parameters.get("device", "auto"), predecessor=predecessor, slice_seconds=limits.get("seconds", 14400.),
            checkpoint_margin_seconds=limits.get("margin_seconds", 120.), max_batches=limits.get("max_batches"))
        artifact = (predecessor if predecessor is not None and predecessor.complete else
                    run_fold(fit_config, endpoint.outer, parameters["seed"], geography=endpoint.geography, variant=variant))
        controller = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]
        export_continuation(artifact, root / "continuation.tar", binding=binding,
                            task_id=task["id"], chain=chain, allowed_root=root)
        progress = {"complete": artifact.complete, "state": "completed" if artifact.complete else "checkpointed",
            "position": controller["position"], "reason": artifact.checkpoint.reason,
            "counts": controller["counts"], "compute": controller["compute"],
            "endpoint": spec.endpoint, "target_id": spec.target_id, "fold": artifact.fold, "seed": artifact.seed,
            "model_hash": artifact.checkpoint.lineage.model_hash, "binding": binding}
        atomic_json(root, "progress.json", progress)
        names = [("continuation.tar", "checkpoint"), ("progress.json", "progress")]
        if artifact.complete:
            held_view = subset(endpoint.data.covariates(tuple(n for n, _ in endpoint.feature_kinds)),
                               artifact.prediction_inputs.original_ids)
            nuisances = predict(artifact, held_view, endpoint.policy)
            frame = pd.DataFrame({key: getattr(nuisances, key) for key in
                ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")})
            frame.to_parquet(root / "nuisances.parquet", index=False)
            # Same complete procedure archive, including frozen models and calibration.
            with (root / "model_bundle.tar").open("xb") as output:
                output.write((root / "continuation.tar").read_bytes())
            atomic_json(root, "metrics.json", {"counts": controller["counts"], "compute": controller["compute"],
                "selection": controller["selection"], "calibration": controller["calibration"],
                "transfer": controller["transfer_diagnostics"], "nuisance_lineage": nuisances.lineage.to_dict()})
            names.extend((("nuisances.parquet", "oof_nuisances"), ("model_bundle.tar", "model_bundle"),
                          ("metrics.json", "metrics")))
        records = [ArtifactRecord(path=name, kind=kind, sha256=file_hash(root / name),
                                  lineage=nuisances.lineage if kind == "oof_nuisances"
                                  else artifact.checkpoint.lineage) for name, kind in names]
        atomic_json(root, "artifact_manifest.json", {"complete": artifact.complete,
            "request_hash": request.content_hash, "artifacts": [record.to_dict() for record in records]})
        records.append(ArtifactRecord(path="artifact_manifest.json", kind="artifact_manifest",
            sha256=file_hash(root / "artifact_manifest.json"), lineage=artifact.checkpoint.lineage))
        result = StageResult(request_hash=request.content_hash, status="pass", artifacts=tuple(records),
            message="Nested procedure complete" if artifact.complete else "Slice checkpointed; nested procedure incomplete")
        result.verify(request)
        return result
    except (ContractError, KeyError, TypeError, ValueError, OSError, tarfile.TarError, RuntimeError) as exc:
        return StageResult(request_hash=request.content_hash, status="fail", artifacts=(), message=str(exc) or type(exc).__name__)
