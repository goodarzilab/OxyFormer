"""Exploratory Q-only screen of saved continuous-endpoint simulation fits."""
import argparse
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, norm, t

from oxyformer.data.loaders import join_outcomes
from oxyformer.provenance import require
from oxyformer.validation import attribution_audit as audit
from oxyformer.validation.coverage import bind_observations, fixed_exposures
from oxyformer.validation.generators import ObservedRecords, TruthArtifact, generate_suite_a
from oxyformer.validation.oracle import ObservedLawOracle

SCHEMA = 'oxyformer.q_parts.v1'
TOP_KEYS = {'schema', 'chain_dir', 'frame_hash', 'variant', 'study', 'code_commit', 'parts'}
PART_KEYS = {'fold', 'seed', 'original_ids', 'mu_a', 'mu_d'}
DEFAULTS = dict(interval_level=98.75, bias_floor=.003, bias_ratio=.5,
                contrast_reduction=.25, iid_coverage_min=16)


def non_q_hash(endpoint, parts):
    records = []
    for part in parts:
        record = json.loads(part.to_json())
        for key in ('mu_a', 'mu_d'):
            del record['payload'][key]
        records.append(record)
    value = dict(parts=records, policy=endpoint.policy.content_hash,
                 targets=endpoint.outer.content_hash, data=endpoint.data.content_hash)
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def part_key(part):
    require(len(set(part.fold_ids)) == len(set(part.seed_ids)) == 1,
            'saved part must contain one fold and seed')
    return part.fold_ids[0], part.seed_ids[0]


def splice(path, chain_dir, frame_hash, endpoint, parts):
    """Validate the companion schema before replacing exactly two arrays."""
    chain = str(Path(chain_dir).resolve())
    def check(ok, fold, key):
        require(ok, f'chain {chain}; fold {fold}; key {key}')
    def keys(value, expected, fold):
        check(isinstance(value, dict), fold, 'object')
        check(set(value) == expected, fold,
              ','.join(sorted(set(value) ^ expected)))
    payload = json.loads(Path(path).read_text())
    keys(payload, TOP_KEYS, '*')
    check(payload['schema'] == SCHEMA, '*', 'schema')
    check(payload['chain_dir'] == chain, '*', 'chain_dir')
    check(payload['frame_hash'] == frame_hash, '*', 'frame_hash')
    for name, cls in (('variant', str), ('study', dict), ('code_commit', str), ('parts', list)):
        check(isinstance(payload[name], cls), '*', name)
    check(bool(payload['variant']), '*', 'variant')
    saved = {part_key(p): p for p in parts}
    check(len(saved) == len(parts), '*', 'parts duplicate saved fold/seed')
    found = {}
    for entry in payload['parts']:
        fold = entry.get('fold', '?') if isinstance(entry, dict) else '?'
        keys(entry, PART_KEYS, fold)
        check(type(entry['fold']) is int, fold, 'fold')
        check(type(entry['seed']) is int, fold, 'seed')
        key = entry['fold'], entry['seed']
        check(key in saved and key not in found, fold, 'parts fold/seed extra or duplicate')
        original = saved[key]
        check(entry['original_ids'] == list(original.original_ids), fold, 'original_ids')
        changes = {}
        for name in ('mu_a', 'mu_d'):
            values = entry[name]
            check(isinstance(values, list) and len(values) == len(original.original_ids), fold, name+' length')
            check(all(type(v) in (int, float) and np.isfinite(v) for v in values), fold, name+' nonfinite or nonnumeric')
            changes[name] = tuple(float(v) for v in values)
        found[key] = replace(original, **changes)
    for key in saved:
        check(key in found, key[0], 'parts missing fold/seed')
    new = [found[part_key(p)] for p in parts]
    before, after = non_q_hash(endpoint, parts), non_q_hash(endpoint, new)
    check(before == after, '*', 'non_Q_hash')
    return new, dict(non_q_before=before, non_q_after=after,
                     q_parts_sha256=sha256(Path(path).read_bytes()).hexdigest(),
                     variant=payload['variant'], study=payload['study'], code_commit=payload['code_commit'])


def load(path):
    result, states, endpoint, recipe = audit._load(path, ('diagnostic', 'm2_pilot_d'))
    require(endpoint.family == 'identity', 'identity-link continuous endpoint required')
    require(len(states) == 1, f'{path}: exactly one state required')
    require(all(len(set(p.seed_ids)) == 1 for p in states[0]['parts']), 'one seed per part required')
    require(len({s for p in states[0]['parts'] for s in p.seed_ids}) == 1, 'one seed required')
    return result, states, endpoint, recipe


def contrast_metrics(d, d0, mask):
    d, d0 = np.asarray(d)[mask], np.asarray(d0)[mask]
    if not len(d):
        return dict(n=0, bias=None, rmse=None, correlation=None, slope=None, sign_agreement=None)
    centered, reference = d-d.mean(), d0-d0.mean()
    denominator = float(reference @ reference)
    norm_product = float(np.linalg.norm(centered)*np.linalg.norm(reference))
    return dict(n=len(d), bias=float(np.mean(d-d0)), rmse=float(np.sqrt(np.mean((d-d0)**2))),
                correlation=float(centered @ reference/norm_product) if norm_product else None,
                slope=float(centered @ reference/denominator) if denominator else None,
                sign_agreement=float(np.mean(np.sign(d) == np.sign(d0))))


def diagnostics(endpoint, parts, oracle):
    split, n, weights = audit.assemble(endpoint, parts)
    require(len(split.seed_ids) == 1, 'one seed required')
    y = np.asarray(join_outcomes(n, endpoint.data, split, n.spec))
    mu, md, r = map(np.asarray, (n.mu_a, n.mu_d, n.r_a))
    q = np.asarray(n.origin_weights); q = q/q.sum()
    true = oracle.nuisances(n.original_ids)
    ma0, md0, r0 = map(np.asarray, (true.mu_a, true.mu_d, true.r_a))
    frame_rows = [oracle.rows[i] for i in n.original_ids]
    movers = np.asarray(endpoint.policy.shift_mask(
        [oracle.observed.a[i] for i in frame_rows], [oracle.frame.support_keys[i] for i in frame_rows]), dtype=bool)
    residual = (y-mu)**2
    weighted = q**2*(r-1)**2
    total = float(np.sum((y-y.mean())**2))
    fit = dict(r2=float(1-residual.sum()/total) if total else None,
               rmse=float(np.sqrt(residual.mean())),
               weighted_mse=float(weighted @ residual/weighted.sum()) if weighted.sum() else None)
    tmle = audit.cv_tmle(n, endpoint.data, weights, 'identity', n.spec, split=split, policy=endpoint.policy)
    targeted = tmle.by_seed[0][1]
    order = {oid: i for i, oid in enumerate(split.original_ids)}
    ix = [order[i] for i in n.original_ids]
    qa, qd = np.asarray(targeted.mu_a)[ix], np.asarray(targeted.mu_d)[ix]
    decompositions = {}
    for method, a, d in (('one_step', mu, md), ('cv_tmle', qa, qd)):
        plugin, correction = d-a, (r-1)*(y-a)
        def decompose(mask):
            w = q[mask]; mass = float(w.sum())
            return dict(target_mass=mass, plugin=float(w @ plugin[mask]/mass),
                        correction=float(w @ correction[mask]/mass)) if mass else dict(target_mass=0., plugin=None, correction=None)
        decompositions[method] = dict(overall=decompose(np.ones(len(y), dtype=bool)),
            folds={str(f): decompose(np.asarray(n.fold_ids) == f) for f in sorted(set(n.fold_ids))})
    return dict(factual=fit, mu_a_rmse=float(np.sqrt(np.mean((mu-ma0)**2))),
                contrast=dict(movers=contrast_metrics(md-mu, md0-ma0, movers),
                              all=contrast_metrics(md-mu, md0-ma0, np.ones(len(y), dtype=bool)),
                              nonmovers=contrast_metrics(md-mu, md0-ma0, ~movers)),
                remainder=float(q @ ((r0-r)*(mu-ma0))), decomposition=decompositions)


def evaluate(endpoint, parts, oracle, locations, psi, spatial_key):
    estimates = audit.replay(endpoint, parts, oracle, locations, cells=('fitted',))['fitted']
    for value in estimates.values():
        error = value['estimate']['value']-psi
        value['error'] = error
        value['coverage'] = {label: bool(abs(error) <= norm.ppf(.975)*value['se'][key]['value'])
                             for label, key in (('iid', 'iid'), ('spatial', spatial_key))}
    return dict(estimates=estimates, psi0=psi, diagnostics=diagnostics(endpoint, parts, oracle))


def screen_chain(chain_dir, baseline, candidates):
    start = time.monotonic()
    result, states, base, recipe = load(chain_dir)
    state = states[0]; record = state['record']
    require(record in result['records'] and record['status'] == 'success', 'no successful record')
    obs = ObservedRecords.from_json(state['sample']['observations'])
    truth = TruthArtifact.from_json(state['sample']['structural_causal_truth'])
    require(obs.frame.content_hash == state['binding']['frame_hash'], 'frame_hash mismatch')
    oracle = ObservedLawOracle(obs, truth, base.policy)
    oracle.nuisances = lru_cache(maxsize=None)(oracle.nuisances)
    endpoint = bind_observations(base, obs)
    parts = state['parts']
    psi = generate_suite_a(obs.frame, oracle.config, base.policy, seed=record['draw']['seed'],
                           fixed_exposures=fixed_exposures(base, obs.frame)).structural_causal_truth.value
    locations = {k: tuple(v) for k, v in recipe['inference']['county_locations'].items()}
    # The saved primary SE identifies the primary registered bandwidth, without a new SE rule.
    saved = record['estimates']['cv_tmle']
    matching = [float(b) for b, se in saved['spatial_se'].items() if se == saved['se']]
    # Pilot records can use an iid gate SE; the inference recipe still names its primary bandwidth.
    bandwidth = recipe['inference']['primary_bandwidth_km']
    spatial_key = f'spatial_{float(bandwidth):g}_km'
    require(not matching or float(bandwidth) in matching, 'primary spatial bandwidth mismatch')
    cells = {'fitted': evaluate(endpoint, parts, oracle, locations, psi, spatial_key)}
    for method, saved_est in record['estimates'].items():
        require(abs(cells['fitted']['estimates'][method]['estimate']['value']-saved_est['value']) < 1e-12,
                f'{chain_dir}: saved {method} estimate mismatch')
        for b, se in saved_est['spatial_se'].items():
            require(abs(cells['fitted']['estimates'][method]['se'][f'spatial_{float(b):g}_km']['value']-se) < 1e-12,
                    f'{chain_dir}: saved spatial SE mismatch')
    oracle_parts = []
    for p in parts:
        true = oracle.nuisances(p.original_ids)
        oracle_parts.append(replace(p, mu_a=true.mu_a, mu_d=true.mu_d))
    cells['oracle_Q'] = evaluate(endpoint, oracle_parts, oracle, locations, psi, spatial_key)
    require(not ({'fitted', 'oracle_Q', 'baseline'} & set(candidates)), 'reserved candidate name')
    for name, path in {'baseline': baseline, **candidates}.items():
        new, provenance = splice(path, chain_dir, state['binding']['frame_hash'], endpoint, parts)
        cells[name] = evaluate(endpoint, new, oracle, locations, psi, spatial_key)
        cells[name]['provenance'] = provenance
    return dict(chain_dir=str(Path(chain_dir).resolve()), repetition_id=record['draw']['repetition_id'],
                campaign_mode=result['campaign_mode'], recipe=recipe.get('study', {}), policy_delta=base.policy.delta_mmhg,
                primary_spatial_key=spatial_key, psi0=psi, cells=cells, wall_seconds=time.monotonic()-start)


def clopper_pearson(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]


def summarize(rows, **options):
    thresholds = {**DEFAULTS, **options}
    require(set(thresholds) == set(DEFAULTS), 'unknown thresholds')
    require(all(type(v) in (int, float) and np.isfinite(v) for v in thresholds.values()), 'nonfinite thresholds')
    require(0 < thresholds['interval_level'] < 100 and thresholds['bias_floor'] > 0
            and 0 <= thresholds['bias_ratio'] <= 1 and 0 <= thresholds['contrast_reduction'] <= 1,
            'invalid thresholds')
    require(type(thresholds['iid_coverage_min']) is int and thresholds['iid_coverage_min'] >= 0, 'invalid iid_coverage_min')
    require(2 <= len(rows) <= 25, 'screen requires 2..25 chains')
    require(len({r['chain_dir'] for r in rows}) == len(rows) and
            len({r['repetition_id'] for r in rows}) == len(rows), 'duplicate chain/repetition')
    names = set(rows[0]['cells'])
    require(all(set(r['cells']) == names for r in rows) and {'fitted', 'oracle_Q', 'baseline'} <= names, 'unpaired cells')
    require(len({r['campaign_mode'] for r in rows}) == 1, 'do not pool primary and pilot chains')
    for name in names - {'fitted', 'oracle_Q'}:
        provenance = [r['cells'][name].get('provenance') for r in rows]
        if any(p is not None for p in provenance):
            require(all(p is not None for p in provenance), 'missing candidate provenance')
            identities = {json.dumps({k: p[k] for k in ('variant', 'study', 'code_commit')}, sort_keys=True) for p in provenance}
            require(len(identities) == 1, f'inconsistent candidate provenance: {name}')
    if all('policy_delta' in r for r in rows):
        require(len({r['policy_delta'] for r in rows}) == 1, 'mixed policy deltas')
    def series(name):
        cells = [r['cells'][name] for r in rows]
        return cells, np.array([c['estimates']['cv_tmle']['estimate']['value'] for c in cells])
    base, est0 = series('baseline')
    psi = np.asarray([r['psi0'] for r in rows]); b0 = float(np.mean(est0-psi))
    s = float(np.sign(-b0))
    rmse0 = float(np.mean([c['diagnostics']['contrast']['movers']['rmse'] for c in base]))
    require(np.isfinite(rmse0) and rmse0 > 0, 'baseline mover RMSE must be positive')
    summaries = {}
    for name in sorted(names):
        cells, est = series(name)
        bc = float(np.mean(est-psi)); differences = s*(est-est0)
        mean_d = float(differences.mean())
        half = float(t.ppf((1+thresholds['interval_level']/100)/2, len(rows)-1)*differences.std(ddof=1)/np.sqrt(len(rows)))
        reduction = 1-float(np.mean([c['diagnostics']['contrast']['movers']['rmse'] for c in cells]))/rmse0
        coverage = {}
        for kind in ('iid', 'spatial'):
            k = sum(c['estimates']['cv_tmle']['coverage'][kind] for c in cells)
            coverage[kind] = dict(count=k, n=len(rows), interval=clopper_pearson(k, len(rows)))
        failed = []
        if abs(bc) > thresholds['bias_ratio']*abs(b0): failed.append('bias_ratio')
        if mean_d-half <= 0: failed.append('paired_interval')
        if reduction < thresholds['contrast_reduction']: failed.append('contrast_reduction')
        if coverage['iid']['count'] < thresholds['iid_coverage_min']: failed.append('iid_coverage_min')
        sd = float(est.std(ddof=1))
        mean_ses = {key: float(np.mean([c['estimates']['cv_tmle']['se'][key]['value'] for c in cells]))
                    for key in cells[0]['estimates']['cv_tmle']['se']}
        summaries[name] = dict(B_c=bc, bias_ratio=abs(bc)/abs(b0) if b0 else None,
            d=differences.tolist(), mean_d=mean_d, paired_interval=[mean_d-half, mean_d+half],
            empirical_sd=sd, mean_se=mean_ses,
            mean_se_over_sd={k: v/sd if sd else None for k,v in mean_ses.items()}, coverage=coverage,
            mover_rmse_reduction=reduction, failing_criteria=failed,
            verdict='floor_not_met' if abs(b0) < thresholds['bias_floor'] else 'fail' if failed else 'pass',
            gross_futility=bool(mean_d <= 0 and reduction < 0))
    return dict(label='exploratory; certifies nothing', n=len(rows), B_0=b0, s=s,
                thresholds=thresholds, cells=summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='JSON list of {chain_dir, baseline, candidates: {name: q_parts_path}}')
    parser.add_argument('--output-dir', required=True, type=Path)
    for name, value in DEFAULTS.items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args(argv)
    entries = json.loads(args.manifest.read_text())
    require(isinstance(entries, list) and 2 <= len(entries) <= 25, 'manifest requires 2..25 chains')
    rows = [screen_chain(**entry) for entry in entries]
    summary = summarize(rows, **{k: getattr(args, k) for k in DEFAULTS})
    report = dict(label='exploratory; certifies nothing', chains=rows, summary=summary)
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'q_screen.json').write_text(text)
    (args.output_dir/'q_screen.md').write_text('# Q screen (exploratory)\n\n```json\n'+text+'```\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
