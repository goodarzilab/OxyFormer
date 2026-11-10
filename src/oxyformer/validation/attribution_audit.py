import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.stats import t

from oxyformer.contracts import OOFNuisances, StageRequest
from oxyformer.data.loaders import join_outcomes
from oxyformer.estimation.covariance import (
    align_estimates, spatial_sensitivities, cluster_covariance,
)
from oxyformer.estimation.mtp import one_step
from oxyformer.estimation.targeting import cv_tmle
from oxyformer.provenance import ContractError, require
from oxyformer.training.nested_cv import PreparedEndpoint
from oxyformer.validation.coverage import bind_observations, fixed_exposures, digest
from oxyformer.validation.generators import ObservedRecords, TruthArtifact, generate_suite_a
from oxyformer.validation.oracle import ObservedLawOracle

FIELDS = ('original_ids', 'fold_ids', 'seed_ids', 'mu_a', 'mu_d', 'r_a', 'r_d', 'origin_weights')
CELLS = ('fitted', 'oracle', 'oracle_Q', 'oracle_g')
RULES = {
    'truth': '|psi0_frame - 0.0178| > 0.0005 for any repetition',
    'branch i': '|mean (Q0,g0) - psi0_frame| <= 0.002 AND mean paired improvement >= 0.007 with its 95% interval excluding zero',
    'branch ii': '|mean (Q0,g0) - psi0_frame| >= 0.75 |D| AND the paired improvement\'s 95% interval includes zero',
    'otherwise': 'inconclusive (partial recovery)',
}

def num(value, basis):
    require(value is None or np.isfinite(value), 'nonfinite num')
    return {'value': value, 'basis': basis}

def paired(differences):
    n = len(differences)
    mean = float(np.mean(differences)) if n else None
    half = float(t.ppf(.975, n-1)*np.std(differences, ddof=1)/np.sqrt(n)) if n > 1 else None
    return {'n': num(n, 'paired repetitions'),
            'mean': num(mean, 'paired oracle-fitted'),
            'lower': num(None if half is None else mean-half, 'paired t interval, 95%'),
            'upper': num(None if half is None else mean+half, 'paired t interval, 95%')}

def decision(truths, fitted, oracle):
    require(len(truths) == len(fitted) == len(oracle) > 0, 'unpaired inputs')
    psi, fit, ora = map(lambda a: float(np.mean(a)), (truths, fitted, oracle))
    stats = paired(np.asarray(oracle)-np.asarray(fitted))
    lo, hi, improvement = (stats[k]['value'] for k in ('lower', 'upper', 'mean'))
    excludes = lo is not None and (lo > 0 or hi < 0)
    includes = lo is not None and lo <= 0 <= hi
    first = abs(ora-psi) <= .002 and improvement >= .007 and excludes
    second = abs(ora-psi) >= .75*abs(psi-fit) and includes
    return {'rules': RULES, 'truth_flagged': any(abs(v-.0178) > .0005 for v in truths),
            'branch_i': first, 'branch_ii': second,
            'verdict': 'branch i' if first else 'branch ii' if second else RULES['otherwise'],
            'psi0_frame': num(psi, 'mean causal truth'),
            'mean_fitted': num(fit, 'paired; fitted'),
            'mean_oracle': num(ora, 'paired; Q0,g0'),
            'D': num(psi-fit, 'mean truth - fitted'), 'paired': stats,
            'interval_note': 'paired t' if lo is not None else 'inconclusive: n < 2'}

def replay(endpoint, parts, oracle, locations):
    seeds = tuple(dict.fromkeys(seed for part in parts for seed in part.seed_ids))
    split = replace(endpoint.outer, seed_ids=seeds)
    lineage = replace(parts[0].lineage, unit_ids=split.original_ids, split_hash=split.content_hash,
                      parent_hashes=(endpoint.data.manifest.content_hash, *(p.content_hash for p in parts)),
                      model_hash=None, seed=None, parameter_count=None)
    fitted = OOFNuisances(spec=endpoint.data.manifest.spec, lineage=lineage,
                         **{n: tuple(v for p in parts for v in getattr(p, n)) for n in FIELDS})
    weights = dict(zip(fitted.original_ids, fitted.origin_weights))
    true = oracle.nuisances(fitted.original_ids)
    by_id = {r.original_id: r for r in endpoint.geography.rows}
    groups = {i: by_id[i].county for i in split.original_ids}
    results = {}
    for cell in CELLS:
        changes = {}
        if cell in ('oracle', 'oracle_Q'):
            changes.update(mu_a=true.mu_a, mu_d=true.mu_d)
        if cell in ('oracle', 'oracle_g'):
            changes.update(r_a=true.r_a, r_d=true.r_d)
        n = replace(fitted, **changes)
        common = dict(split=split, policy=endpoint.policy)
        tmle = cv_tmle(n, endpoint.data, weights, 'identity', fitted.spec, **common)
        estimates = {'one_step': one_step(n, endpoint.data, weights, fitted.spec, **common),
                     'cv_tmle': tmle.estimate}
        influence = align_estimates(estimates)
        spatial = spatial_sensitivities(influence, groups, locations, interpretation='geographic_process')
        cluster = cluster_covariance(influence, groups, interpretation='geographic_process')
        y = np.asarray(join_outcomes(n, endpoint.data, split, fitted.spec))
        rows = {(i, seed): j for j, (i, seed) in enumerate(zip(n.original_ids, n.seed_ids))}
        q = np.array([weights[i] for i in split.original_ids]); q /= q.sum()
        results[cell] = {}
        for column, (m, e) in enumerate(estimates.items()):
            errors = []
            for seed, fit in tmle.by_seed:
                ix = [rows[i, seed] for i in split.original_ids]
                mu = np.array(fit.mu_a if m == 'cv_tmle' else np.asarray(n.mu_a)[ix])
                errors.append((np.asarray(n.r_a)[ix]-1)*(y[ix]-mu))
            u = q*np.mean(errors, axis=0)
            se = {f'spatial_{b:g}_km': cov.standard_errors[column] for b, cov in spatial.items()}
            se['county_0_km'] = cluster.standard_errors[column]
            se['iid'] = np.sqrt(len(q)/(len(q)-1)*np.sum(np.asarray(e.influence)**2))
            se['frame_conditional'] = np.sqrt(np.sum(u**2))
            results[cell][m] = {'estimate': num(e.value, f'{cell}; {m}'),
                'se': {name: num(float(value), 'residual-only diagonal' if name == 'frame_conditional'
                                   else name) for name, value in se.items()}}
    return results

def load(path):
    path = Path(path)
    result = json.loads((path/'result.json').read_text())
    require(result['terminal'] and result['campaign_mode'] == 'diagnostic', 'terminal diagnostic required')
    with tarfile.open(path/'continuation.tar', 'r:') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members) and all(m.isfile() for m in members), 'invalid tar members')
        header = json.load(archive.extractfile('coverage.json'))
        def blob(key):
            require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'invalid blob key')
            value = archive.extractfile('blobs/'+key).read()
            require(sha256(value).hexdigest() == key, 'blob digest mismatch')
            return value.decode()
        states = [{**state, 'parts': [OOFNuisances.from_json(blob(k)) for k in state['parts']],
                   'sample': {k: blob(state['sample'][k]) for k in ('observations', 'structural_causal_truth')}} for state in header['states']]
    request = StageRequest.from_json((path/'_execution/request.json').read_text())
    def dependency(expected, cls=None):
        for name, hashed in zip(request.dependency_paths, request.dependency_hashes):
            if hashed == expected:
                source = Path(name)
                # Only rebase paths explicitly anchored inside the saved attempt.
                if not source.exists() and source.is_relative_to(request.output_dir):
                    source = path/source.relative_to(request.output_dir)
                raw = source.read_bytes()
                require(sha256(raw).hexdigest() == expected, 'dependency digest mismatch')
                return cls.from_json(raw.decode()) if cls else json.loads(raw)
        raise ContractError('missing dependency: '+expected)
    endpoint = dependency(header['binding']['endpoint_hash'], PreparedEndpoint)
    recipe = dependency(header['binding']['lock_hash'])['recipe']
    require(digest(recipe) == header['binding']['recipe_hash'], 'recipe digest mismatch')
    require(header['binding']['frame_hash'] == states[0]['binding']['frame_hash'], 'frame binding mismatch')
    return result, states, endpoint, recipe

def audit(path):
    result, states, base, recipe = load(path)
    output = []
    for state in states:
        record = state['record']
        require(record in result['records'] and record['status'] == 'success', 'no successful record')
        sample = state['sample']
        obs = ObservedRecords.from_json(sample['observations'])
        truth = TruthArtifact.from_json(sample['structural_causal_truth'])
        require(obs.frame.content_hash == state['binding']['frame_hash'], 'frame hash mismatch')
        oracle = ObservedLawOracle(obs, truth, base.policy)
        endpoint = bind_observations(base, obs)
        locations = {k: tuple(v) for k, v in recipe['inference']['county_locations'].items()}
        cells = replay(endpoint, state['parts'], oracle, locations)
        recomputed = generate_suite_a(obs.frame, oracle.config, base.policy,
            seed=record['draw']['seed'], fixed_exposures=fixed_exposures(base, obs.frame))
        psi = recomputed.structural_causal_truth.value
        for m, saved in record['estimates'].items():
            for bandwidth, saved_se in saved['spatial_se'].items():
                require(abs(cells['fitted'][m]['se'][f'spatial_{float(bandwidth):g}_km']['value']-saved_se) < 1e-12, 'spatial SE mismatch')
            require(abs(cells['fitted'][m]['estimate']['value']-saved['value']) < 1e-12,
                    'estimate mismatch')
        output.append({'repetition_id': record['draw']['repetition_id'], 'cells': cells,
            'excluded_fixed_exposure_rows': num(oracle.excluded_fixed_exposure_rows, 'zero weight; empty frozen support'),
            'psi0_frame': num(psi, 'fixed-frame integration'),
            'saved_causal_truth': num(record['causal_truth'], 'saved causal truth'),
            'truth_difference': num(psi-record['causal_truth'], 'recomputed - saved truth')})
    return output

def summarize(rows):
    summary = {}
    for m in ('cv_tmle', 'one_step'):
        def entries(cell):
            return [r['cells'][cell][m] for r in rows]
        def values(cell):
            return [e['estimate']['value'] for e in entries(cell)]
        summary[m] = {'decision': decision([r['psi0_frame']['value'] for r in rows],
            values('fitted'), values('oracle')), 'cells': {}}
        for cell in CELLS:
            series = entries(cell)
            sd = float(np.std(values(cell), ddof=1)) if len(rows)>1 else None
            ratios = {}
            for name in series[0]['se']:
                mean_se = float(np.mean([e['se'][name]['value'] for e in series]))
                ratios[name] = {'mean_se': num(mean_se, f'{name}; repetition mean'),
                    'sd': num(sd, 'repetition SD, ddof=1'),
                    'mean_se_over_sd': num(mean_se/sd if sd else None, 'mean_se / SD; descriptive')}
            summary[m]['cells'][cell] = {'se_summary': ratios, 'paired': paired(np.subtract(values(cell), values('fitted')))}
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempts', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    rows, skipped, seen = [], [], set()
    for path in args.attempts:
        try:
            found = audit(path)
            require(all(r['repetition_id'] not in seen for r in found), 'duplicate repetition')
            rows.extend(found); seen.update(r['repetition_id'] for r in found)
        except (OSError, KeyError, ValueError, TypeError, tarfile.TarError, ContractError) as exc:
            skipped.append({'attempt': str(path), 'status': 'unreadable: '+str(exc)})
    report = {'label': 'diagnostic only; no release certification', 'repetitions': rows,
              'skipped': skipped, 'summary': summarize(rows) if rows else {},
              'rule_basis': 'design memo A3'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.output_dir/'attribution_audit.json').write_text(text+'\n')
    markdown = '# Attribution audit (diagnostic only)\n\n```json\n'+text+'\n```\n'
    (args.output_dir/'attribution_audit.md').write_text(markdown)
    print(markdown)
    return 0 if rows else 1

if __name__ == '__main__':
    raise SystemExit(main())
