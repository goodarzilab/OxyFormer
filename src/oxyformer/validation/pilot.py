"""Bounded paired diagnostic pilot; source artifacts are read only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.stats import t

from oxyformer.contracts import StageRequest
from oxyformer.provenance import ContractError, file_hash, require
from oxyformer.validation import attribution_audit
from oxyformer.validation.coverage import fixed_exposures, validate_draws, digest
from oxyformer.validation.generators import ObservedRecords, TruthArtifact, generate_suite_a
from oxyformer.validation.scm import SCMConfig

RULE = 'lowest canonical draw indices among collected source chains; declared scenario only'
LABEL = 'diagnostic pilot; no release criterion certified'


def source(path, chain):
    """Use the merged archive reader, and verify every saved blob and source lock."""
    try:
        result, states, endpoint, recipe = attribution_audit.load(path)
        with tarfile.open(Path(path) / 'continuation.tar', 'r:') as archive:
            for member in archive.getmembers():
                if member.name.startswith('blobs/'):
                    require(sha256(archive.extractfile(member).read()).hexdigest() == member.name[6:],
                            'source blob hash mismatch')
            header = json.load(archive.extractfile('coverage.json'))
        request = StageRequest.from_json((Path(path) / '_execution/request.json').read_text())
        return result, states, endpoint, recipe, header, request
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError) as exc:
        raise ContractError('paired source chain ' + str(chain) + ': ' + str(exc)) from exc


def pin_draws(manifest_path, declaration):
    """Select before inspecting estimates. No filtering on completion or outcomes."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest['campaign'] == declaration['paired_from'], 'paired manifest campaign mismatch')
    candidates = []
    seen_chains = set()
    for entry in manifest['chains']:
        chain = entry['chain']
        require(chain not in seen_chains, 'duplicate paired source chain ' + chain)
        seen_chains.add(chain)
        path = Path(entry['terminal_leaf_attempt_dir'])
        try:
            result = json.loads((path / 'result.json').read_text())
            for record in result['records']:
                draw = record['draw']
                if draw['scenario'] in declaration['scenarios']:
                    validate_draws([draw])
                    candidates.append((draw['index'], chain, path, draw))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ContractError('paired source chain ' + chain + ': ' + str(exc)) from exc
    require(len({v[0] for v in candidates}) == len(candidates), 'duplicate canonical source draw index')
    count = declaration['repetitions_per_scenario']
    require(len(candidates) >= count, 'insufficient collected paired draws')
    pins = []
    for index, chain, path, draw in sorted(candidates, key=lambda v: v[0])[:count]:
        result, states, base, recipe, header, request = source(path, chain)
        require(request.code_identity == manifest['tree'], 'paired source chain ' + chain + ': source tree mismatch')
        require(header['binding']['lock_hash'] == manifest['lock_hash'],
                'paired source chain ' + chain + ': source lock hash mismatch')
        matches = [state for state in states if state['record'] and state['record']['draw'] == draw]
        require(len(matches) == 1 and matches[0]['record'] in result['records'],
                'paired source chain ' + chain + ': missing pinned draw')
        state = matches[0]
        require(state['record']['status'] == 'success', 'paired source chain ' + chain + ': draw not successful')
        obs = ObservedRecords.from_json(state['sample']['observations'])
        truth = TruthArtifact.from_json(state['sample']['structural_causal_truth'])
        require(SCMConfig.from_json(truth.generator_configuration).name == draw['scenario'],
                'paired source chain ' + chain + ': source scenario mismatch')
        pins.append({'source_campaign': manifest['campaign'], 'source_chain': chain,
            'source_path': str(path.resolve()), 'draw_index': index, 'draw': deepcopy(draw),
            'repetition_id': draw['repetition_id'], 'seed': draw['seed'], 'scenario': draw['scenario'],
            'source_lock_hash': header['binding']['lock_hash'], 'source_tree': request.code_identity,
            'observations_hash': obs.content_hash, 'frame_hash': obs.frame.content_hash,
            'truth_hash': truth.content_hash, 'source_result_hash': file_hash(path / 'result.json'),
            'source_archive_hash': file_hash(path / 'continuation.tar'), 'policy_hash': base.policy.content_hash,
            'scm_hash': truth.config_hash, 'selection_rule': RULE,
            'manifest_hash': file_hash(manifest_path)})
    validate_draws([pin['draw'] for pin in pins])
    return pins


def validate_pins(lock):
    from oxyformer.validation.campaign import pilot_declaration
    d = pilot_declaration(lock['recipe'])
    pins = lock.get('paired_draws', [])
    require(len(pins) == d['repetitions_per_scenario'], 'pilot pinned draw count drift')
    draws = [draw for batch in lock['batches'] for draw in batch['draws']]
    require(draws == [pin['draw'] for pin in pins], 'pilot pinned batch draws drift')
    require([p['draw_index'] for p in pins] == sorted({p['draw_index'] for p in pins}),
            'pilot canonical draw order drift')
    for pin in pins:
        require(pin['source_campaign'] == d['paired_from'] and pin['selection_rule'] == RULE,
                'pilot paired source specification drift')
        require(all(pin[key] == pin['draw'][key] for key in ('seed', 'repetition_id', 'scenario'))
                and pin['draw_index'] == pin['draw']['index'], 'pilot pinned identity drift')


def pinned_source(pin):
    path, chain = Path(pin['source_path']), pin['source_chain']
    try:
        require(file_hash(path / 'result.json') == pin['source_result_hash'], 'source result hash mismatch')
        require(file_hash(path / 'continuation.tar') == pin['source_archive_hash'], 'source archive hash mismatch')
        result, states, base, recipe, header, request = source(path, chain)
        require(header['binding']['lock_hash'] == pin['source_lock_hash']
                and request.code_identity == pin['source_tree'], 'source lock/tree mismatch')
        matches = [s for s in states if s['record'] and s['record']['draw'] == pin['draw']]
        require(len(matches) == 1, 'missing pinned draw')
        state = matches[0]
        obs = ObservedRecords.from_json(state['sample']['observations'])
        truth = TruthArtifact.from_json(state['sample']['structural_causal_truth'])
        require(obs.content_hash == pin['observations_hash'] and obs.frame.content_hash == pin['frame_hash']
                and truth.content_hash == pin['truth_hash'], 'pinned observations/frame/truth mismatch')
        require(base.policy.content_hash == pin['policy_hash'] and truth.config_hash == pin['scm_hash'],
                'pinned policy/SCM mismatch')
        return state, obs, truth, base, recipe
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError) as exc:
        raise ContractError('paired source chain ' + chain + ': ' + str(exc)) from exc


def verify_regeneration(lock, draws=None, *, template=None, frame=None, scenario=None):
    """Falsification check used by public admission and each chain's first leaf."""
    validate_pins(lock)
    pins = lock['paired_draws']
    if draws is not None:
        pins = [pin for pin in pins if pin['draw'] in draws]
        require(len(pins) == len(draws), 'pilot leaf draw not pinned')
    for pin in pins:
        state, obs, truth, base, recipe = pinned_source(pin)
        config = SCMConfig.from_json(truth.generator_configuration)
        chain = pin['source_chain']
        require(config.content_hash == pin['scm_hash'], 'paired source chain ' + chain + ': SCM hash mismatch')
        if template is not None:
            require(template.policy.content_hash == base.policy.content_hash and frame.content_hash == obs.frame.content_hash
                    and scenario.content_hash == config.content_hash,
                    'paired source chain ' + chain + ': pilot generation inputs differ from source')
        regenerated = generate_suite_a(obs.frame, config, base.policy, seed=pin['seed'],
                                      fixed_exposures=fixed_exposures(base, obs.frame))
        require(regenerated.observations.content_hash == pin['observations_hash']
                and regenerated.structural_causal_truth.content_hash == pin['truth_hash'],
                'paired source chain ' + chain + ': regeneration falsification failed before fitting')
    return {'verified_repetitions': [pin['repetition_id'] for pin in pins], 'selection_rule': RULE}


def number(value, basis):
    return attribution_audit.num(value, basis)


def interval(values, basis):
    n = len(values)
    mean = float(np.mean(values)) if n else None
    half = float(t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
    return {'n': number(n, 'paired repetitions; exploratory'), 'mean': number(mean, basis),
        'lower': number(None if half is None else mean - half, basis + '; paired t 95% interval'),
        'upper': number(None if half is None else mean + half, basis + '; paired t 95% interval')}


def paired_readout(lock, records):
    validate_pins(lock)
    from oxyformer.validation.campaign import pilot_declaration
    declaration = pilot_declaration(lock["recipe"])
    pilot_cell = "cell " + declaration["cell"]
    by_id = {r['draw']['repetition_id']: r for r in records}
    require(len(by_id) == len(records), 'duplicate pilot readout repetition')
    rows = []
    for pin in lock['paired_draws']:
        if pin['repetition_id'] not in by_id:
            continue  # Local collectors describe their own declared pairs only.
        record = by_id[pin['repetition_id']]
        require(record['draw'] == pin['draw'], 'pilot readout draw mismatch')
        state, obs, truth, base, recipe = pinned_source(pin)
        from oxyformer.validation.coverage import bind_observations
        from oxyformer.validation.oracle import ObservedLawOracle
        cells = attribution_audit.replay(bind_observations(base, obs), state['parts'],
            ObservedLawOracle(obs, truth, base.policy), recipe['inference']['county_locations'])
        row = {'repetition_id': pin['repetition_id'], 'source_chain': pin['source_chain'],
            'seed': number(pin['seed'], 'pinned source simulation seed'),
            'truth': number(truth.value, 'per-draw saved structural causal truth'), 'methods': {}}
        for method in ('one_step', 'cv_tmle'):
            source_estimate = state['record']['estimates'][method]['value']
            estimate = record.get('estimates', {}).get(method) if record['status'] == 'success' else None
            fitted = None if estimate is None else estimate['value']
            row['methods'][method] = {'status': record['status'],
                'pilot': number(fitted, pilot_cell + '; ' + method),
                'pilot_se_definitions': {} if estimate is None else {
                    name: {key: number(value, pilot_cell + '; ' + method + '; ' + name) if type(value) in (int, float) else value
                           for key, value in definition.items()}
                    for name, definition in estimate['se_definitions'].items()},
                'source': number(source_estimate, 'cell e; source chain record; ' + method),
                'oracle_cells': {cell: cells[cell][method] for cell in attribution_audit.CELLS if cell != 'fitted'},
                'pilot_minus_source': number(None if fitted is None else fitted - source_estimate,
                                             'paired ' + pilot_cell + ' minus cell e; ' + method),
                'oracle_minus_pilot': {cell: number(None if fitted is None else cells[cell][method]['estimate']['value'] - fitted,
                                            'paired ' + cell + ' minus ' + pilot_cell + '; ' + method)
                                      for cell in attribution_audit.CELLS if cell != 'fitted'}}
        rows.append(row)
    summary = {}
    for method in ('one_step', 'cv_tmle'):
        successful = [row['methods'][method] for row in rows if row['methods'][method]['status'] == 'success']
        summary[method] = {'pilot_minus_source': interval([r['pilot_minus_source']['value'] for r in successful],
                                                        'paired ' + pilot_cell + ' minus cell e; ' + method),
            'oracle_minus_pilot': {cell: interval([r['oracle_minus_pilot'][cell]['value'] for r in successful],
                                                 'paired ' + cell + ' minus ' + pilot_cell + '; ' + method)
                                  for cell in attribution_audit.CELLS if cell != 'fitted'}}
    report = {'label': LABEL, 'certifies_release_criteria': False,
        'lock_hash': digest(lock), 'owner_block': declaration['campaign_mode'],
        'pilot_cell': declaration['cell'], 'gate_se': declaration['gate_se'],
        'statement': 'Planned five pairs (n = 5) are exploratory; descriptive paired t intervals certify nothing. Each collector states its observed pair count. Draws are never pooled away.',
        'pairs': rows, 'summary': summary}
    markdown = '# ' + LABEL + '\n\n' + report['statement'] + '\n\n```json\n' + json.dumps(report, indent=2) + '\n```\n'
    return report, markdown
