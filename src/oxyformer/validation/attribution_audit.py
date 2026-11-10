"""CPU diagnostic replay of saved repetitions; never fits nuisance regressions."""
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

CELLS = ('fitted', 'oracle', 'oracle_Q', 'oracle_g')
RULES = {
    'truth': '|psi0_frame - 0.0178| > 0.0005 for any repetition',
    'branch i': '|mean (Q0,g0) - psi0_frame| <= 0.002 AND mean paired improvement >= 0.007 with its 95% interval excluding zero',
    'branch ii': '|mean (Q0,g0) - psi0_frame| >= 0.75 |D| AND the paired improvement\'s 95% interval includes zero',
    'otherwise': 'inconclusive (partial recovery)',
}


def number(value, basis):
    if isinstance(value, np.generic):
        value = value.item()
    require(value is None or np.isfinite(value), 'nonfinite audit number')
    return {'value': value, 'basis': basis}


def paired_statistics(differences):
    n = len(differences)
    mean = float(np.mean(differences)) if n else None
    half = float(t.ppf(.975, n-1)*np.std(differences, ddof=1)/np.sqrt(n)) if n > 1 else None
    return {'n': number(n, 'distinct paired repetitions'),
            'mean': number(mean, 'paired oracle minus fitted'),
            'lower': number(None if half is None else mean-half, 'paired t interval, 95%'),
            'upper': number(None if half is None else mean+half, 'paired t interval, 95%')}


def decision(truths, fitted, oracle):
    require(len(truths) == len(fitted) == len(oracle) > 0, 'paired decision inputs required')
    psi, fit, ora = map(lambda a: float(np.mean(a)), (truths, fitted, oracle))
    paired = paired_statistics(np.asarray(oracle)-np.asarray(fitted))
    lo, hi, improvement = (paired[k]['value'] for k in ('lower', 'upper', 'mean'))
    excludes = lo is not None and (lo > 0 or hi < 0)
    includes = lo is not None and lo <= 0 <= hi
    first = abs(ora-psi) <= .002 and improvement >= .007 and excludes
    second = abs(ora-psi) >= .75*abs(psi-fit) and includes
    return {'rules': RULES, 'truth_flagged': any(abs(v-.0178) > .0005 for v in truths),
            'branch_i': first, 'branch_ii': second,
            'verdict': 'branch i' if first else 'branch ii' if second else RULES['otherwise'],
            'psi0_frame': number(psi, 'mean recomputed fixed-frame causal truth'),
            'mean_fitted': number(fit, 'paired; fitted nuisances'),
            'mean_oracle': number(ora, 'paired; Q0,g0'),
            'D': number(psi-fit, 'mean recomputed truth minus mean fitted'), 'paired': paired,
            'interval_note': 'paired t interval available' if lo is not None else 'inconclusive: fewer than two repetitions'}


def replay(endpoint, parts, oracle, locations):
    """Reassemble exactly as coverage.estimate_repetition; vary only Q/g fields."""
    names = ('original_ids', 'fold_ids', 'seed_ids', 'mu_a', 'mu_d', 'r_a', 'r_d', 'origin_weights')
    seeds = tuple(dict.fromkeys(seed for part in parts for seed in part.seed_ids))
    split = replace(endpoint.outer, seed_ids=seeds)
    lineage = replace(parts[0].lineage, unit_ids=split.original_ids, split_hash=split.content_hash,
                      parent_hashes=(endpoint.data.manifest.content_hash, *(p.content_hash for p in parts)),
                      model_hash=None, seed=None, parameter_count=None)
    fitted = OOFNuisances(spec=endpoint.data.manifest.spec, lineage=lineage,
                         **{n: tuple(v for p in parts for v in getattr(p, n)) for n in names})
    weights = dict(zip(split.original_ids, [dict(zip(fitted.original_ids, fitted.origin_weights))[i] for i in split.original_ids]))
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
        nuisances = replace(fitted, **changes)
        common = dict(split=split, policy=endpoint.policy)
        tmle = cv_tmle(nuisances, endpoint.data, weights, 'identity', fitted.spec, **common)
        estimates = {'one_step': one_step(nuisances, endpoint.data, weights, fitted.spec, **common),
                     'cv_tmle': tmle.estimate}
        influence = align_estimates(estimates)
        spatial = spatial_sensitivities(influence, groups, locations, interpretation='geographic_process')
        cluster = cluster_covariance(influence, groups, interpretation='geographic_process')
        y = np.asarray(join_outcomes(nuisances, endpoint.data, split, fitted.spec))
        rows = {(i, seed): j for j, (i, seed) in enumerate(zip(nuisances.original_ids, nuisances.seed_ids))}
        q = np.array([weights[i] for i in split.original_ids]); q /= q.sum()
        results[cell] = {}
        for column, (method, estimate) in enumerate(estimates.items()):
            residuals = []
            for seed, fluctuation in tmle.by_seed:
                indices = [rows[i, seed] for i in split.original_ids]
                mu = np.array(fluctuation.mu_a if method == 'cv_tmle' else np.asarray(nuisances.mu_a)[indices])
                residuals.append((np.asarray(nuisances.r_a)[indices]-1)*(y[indices]-mu))
            residual = q*np.mean(residuals, axis=0)
            se = {f'spatial_{b:g}_km': cov.standard_errors[column] for b, cov in spatial.items()}
            se['county_0_km'] = cluster.standard_errors[column]
            se['iid'] = np.sqrt(len(q)/(len(q)-1)*np.sum(np.asarray(estimate.influence)**2))
            se['frame_conditional'] = np.sqrt(np.sum(residual**2))
            results[cell][method] = {'estimate': number(estimate.value, f'{cell}; unchanged {method}'),
                'se': {name: number(float(value), f'{cell}; {method}; {name}; residual only, fixed frame' if name == 'frame_conditional'
                                   else f'{cell}; {method}; {name}') for name, value in se.items()}}
    return results


def load_attempt(path):
    path = Path(path)
    result = json.loads((path/'result.json').read_text())
    require(result['terminal'] and result['campaign_mode'] == 'diagnostic', 'terminal diagnostic leaf required')
    with tarfile.open(path/'continuation.tar', 'r:') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members) and all(m.isfile() for m in members), 'invalid archive members')
        header = json.load(archive.extractfile('coverage.json'))
        def blob(key):
            require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'invalid blob key')
            value = archive.extractfile('blobs/'+key).read()
            require(sha256(value).hexdigest() == key, 'blob digest mismatch')
            return value.decode()
        states = [{**state, 'parts': [OOFNuisances.from_json(blob(k)) for k in state['parts']],
                   'sample': {k: blob(v) for k, v in state['sample'].items()}} for state in header['states']]
    request = StageRequest.from_json((path/'_execution/request.json').read_text())
    endpoint = None
    for name, expected in zip(request.dependency_paths, request.dependency_hashes):
        if expected == header['binding']['endpoint_hash']:
            raw = Path(name).read_bytes()
            require(sha256(raw).hexdigest() == expected, 'endpoint dependency digest mismatch')
            endpoint = PreparedEndpoint.from_json(raw.decode())
            break
    require(endpoint is not None, 'missing bound PreparedEndpoint dependency')
    task_raw = Path(request.task_path).read_text()
    require(sha256(task_raw.encode()).hexdigest() == request.task_hash, 'task digest mismatch')
    recipe = json.loads(task_raw)['parameters'].get('recipe')
    if recipe is None:
        for name, expected in zip(request.dependency_paths, request.dependency_hashes):
            if expected == header['binding']['lock_hash']:
                raw = Path(name).read_bytes()
                require(sha256(raw).hexdigest() == expected, 'lock digest mismatch')
                recipe = json.loads(raw)['recipe']
                break
    require(recipe is not None, 'missing recipe lock dependency')
    require(digest(recipe) == header['binding']['recipe_hash'], 'recipe digest mismatch')
    require(header['binding']['frame_hash'] == states[0]['binding']['frame_hash'], 'frame binding mismatch')
    return result, states, endpoint, recipe


def audit_attempt(path):
    result, states, template, recipe = load_attempt(path)
    output = []
    for state in states:
        record = state['record']
        require(record in result['records'] and record['status'] == 'success', 'missing successful saved record')
        sample = state['sample']
        observations = ObservedRecords.from_json(sample['observations'])
        truth = TruthArtifact.from_json(sample['structural_causal_truth'])
        require(observations.frame.content_hash == state['binding']['frame_hash'], 'frame hash mismatch')
        oracle = ObservedLawOracle(observations, truth, template.policy)
        endpoint = bind_observations(template, observations)
        locations = {k: tuple(v) for k, v in recipe['inference']['county_locations'].items()}
        cells = replay(endpoint, state['parts'], oracle, locations)
        recomputed = generate_suite_a(observations.frame, oracle.config, template.policy,
            seed=record['draw']['seed'], fixed_exposures=fixed_exposures(template, observations.frame))
        psi = recomputed.structural_causal_truth.value
        for method, saved in record['estimates'].items():
            for bandwidth, saved_se in saved['spatial_se'].items():
                require(abs(cells['fitted'][method]['se'][f'spatial_{float(bandwidth):g}_km']['value']-saved_se) < 1e-12, 'fitted spatial SE differs from saved SE')
            require(abs(cells['fitted'][method]['estimate']['value']-saved['value']) < 1e-12,
                    'fitted replay differs from saved estimate')
        output.append({'repetition_id': record['draw']['repetition_id'], 'cells': cells,
            'psi0_frame': number(psi, 'fixed-frame causal integration'),
            'saved_causal_truth': number(record['causal_truth'], 'saved terminal-leaf record'),
            'truth_difference': number(psi-record['causal_truth'], 'recomputed minus saved causal truth')})
    return output


def summarize(rows):
    summary = {}
    for method in ('cv_tmle', 'one_step'):
        summary[method] = {'decision': decision([r['psi0_frame']['value'] for r in rows],
            [r['cells']['fitted'][method]['estimate']['value'] for r in rows],
            [r['cells']['oracle'][method]['estimate']['value'] for r in rows]), 'cells': {}}
        for cell in CELLS:
            entries = [r['cells'][cell][method] for r in rows]
            sd = float(np.std([e['estimate']['value'] for e in entries], ddof=1)) if len(rows)>1 else None
            ratios = {}
            for name in entries[0]['se']:
                mean_se = float(np.mean([e['se'][name]['value'] for e in entries]))
                ratios[name] = {'mean_se': number(mean_se, f'{cell}; {method}; {name}; mean over repetitions'),
                    'sd': number(sd, f'{cell}; {method}; repetition SD, ddof=1'),
                    'mean_se_over_sd': number(mean_se/sd if sd else None, 'coverage.py mean_se / SD; descriptive, no gate')}
            summary[method]['cells'][cell] = {'se_summary': ratios, 'paired': paired_statistics([
                r['cells'][cell][method]['estimate']['value']-r['cells']['fitted'][method]['estimate']['value'] for r in rows])}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempts', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    rows, skipped, seen = [], [], set()
    for path in args.attempts:
        try:
            found = audit_attempt(path)
            require(all(r['repetition_id'] not in seen for r in found), 'duplicate repetition')
            rows.extend(found); seen.update(r['repetition_id'] for r in found)
        except (OSError, KeyError, ValueError, TypeError, tarfile.TarError, ContractError) as exc:
            skipped.append({'attempt': str(path), 'status': 'unreadable: '+str(exc)})
    report = {'label': 'diagnostic only; no release criterion certified', 'repetitions': rows,
              'skipped': skipped, 'summary': summarize(rows) if rows else {},
              'basis_note': 'Every numeric value is wrapped with its computation basis; null means unavailable.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.output_dir/'attribution_audit.json').write_text(text+'\n')
    markdown = '# Attribution audit (diagnostic only)\n\nRules, inputs, verdicts and all SE definitions:\n\n```json\n'+text+'\n```\n'
    (args.output_dir/'attribution_audit.md').write_text(markdown)
    print(markdown)
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
