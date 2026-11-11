"""Bounded paired diagnostic pilot; source artifacts are read only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tarfile

from oxyformer.contracts import StageRequest
from oxyformer.provenance import ContractError, file_hash, require
from oxyformer.validation import attribution_audit
from oxyformer.validation.coverage import fixed_exposures, validate_draws, digest
from oxyformer.validation.generators import ObservedRecords, TruthArtifact, generate_suite_a
from oxyformer.validation.scm import SCMConfig

RULE = 'lowest canonical draw indices among collected source chains; declared scenario only'


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


def validate_pins(lock, approvals=None):
    from oxyformer.validation.campaign import pilot_declaration
    d = pilot_declaration(lock['recipe']) if approvals is None else pilot_declaration(lock['recipe'], approvals)
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


def verify_regeneration(lock, draws=None, *, template=None, frame=None, scenario=None, approvals=None):
    """Falsification check used by public admission and each chain's first leaf."""
    validate_pins(lock, approvals)
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
