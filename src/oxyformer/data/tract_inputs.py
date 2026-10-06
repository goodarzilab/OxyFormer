"""Outcome-free typed tract inputs from the reviewed historical US adapters.

Only the six approved ACS sequences and geography members are staged. The
3.4 GB gzip tar is consumed as a stream; each state's adapter audit is released
before the next state is loaded. Raw outcomes never leave the adapter boundary.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pandas as pd
import yaml

from oxyformer.contracts import (ColumnSpec, DataManifest, EstimandSpec,
                                SourceManifest, source_lineage_hash)
from oxyformer.data.adapters.acs import load_acs
from oxyformer.data.adapters.usaleep import SourceFile, load_usaleep, mapping_hash
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.data.loaders import load_records
from oxyformer.data.source_manifest import load_source
from oxyformer.execution.integrity import open_regular, read_regular, verify_input_hash
from oxyformer.provenance import ArtifactLineage, canonical_json, file_hash, require

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = 'usaleep_life_expectancy'
MAPPING = 'configs/adapters/us.yaml'


def _source_file(path, section, source_id, uri, license_hash):
    return SourceFile(path=path, manifest=SourceManifest(
        source_id=source_id, version=section['release'], uri=uri,
        payload_hash=file_hash(path), license_hash=license_hash,
        schema_hash=mapping_hash(section), field_mapping=tuple(section['field_mapping'].items()),
        mapping_status=section['mapping_status'], mapping_review_id=MAPPING))


def load_us_inputs(payload, receipt_path, scratch, *, source=None, mapping=None, states=None):
    """Return the full footprint ACS frame, USALEEP metadata, and source bindings.

    Injection arguments support small offline fixtures. The registered stage
    uses only repository inventories/mappings; tasks cannot override them.
    """
    source = load_source('us') if source is None else source
    mapping = yaml.safe_load((ROOT / MAPPING).read_text()) if mapping is None else mapping
    states = yaml.safe_load((ROOT / 'configs/exposure.yaml').read_text())['state_fips'] if states is None else states
    states = {s.lower(): fips for s, fips in states.items()}
    receipt = json.loads(read_regular(receipt_path))
    require(receipt.get('status') == 'complete' and receipt.get('manifest_id') == 'us',
            'complete US acquisition required')
    require(receipt['manifest_sha256'] == sha256(canonical_json(source).encode()).hexdigest(),
            'US receipt differs from repository source manifest')
    declared = {r['id']: r for r in source['resources']}
    acquired = {r['id']: r for r in receipt['resources']}
    require(len(acquired) == len(receipt['resources']) and acquired.keys() == declared.keys(),
            'US receipt resource inventory mismatch')
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    required = {'usaleep_data', 'usaleep_terms', 'usaleep_dictionary', 'acs_tracts',
                'acs_geography', 'acs_terms', 'acs_dictionary', 'acs_technical'}
    require(required <= declared.keys(), 'US acquisition missing adapter resources')
    verify_input_hash(payload, receipt['payload_sha256'])
    require(Path(payload).stat().st_size == receipt['payload_bytes'], 'US payload size mismatch')
    with open_regular(payload) as handle, tarfile.open(fileobj=handle, mode='r:') as archive:
        members = {}
        for member in archive:
            require(member.isfile() and member.name not in members, 'duplicate or nonregular US member')
            members[member.name] = member
        require(set(members) == {r['destination'] for r in acquired.values()}, 'US archive inventory mismatch')
        for rid in required:
            entry, spec = acquired[rid], declared[rid]
            require(entry['destination'] == spec['destination'], 'US resource destination mismatch')
            member = members[entry['destination']]
            require(member.size == entry['bytes'] and member.size <= spec['max_bytes'], 'US resource size mismatch')
            require(spec.get('expected_bytes', member.size) == member.size and
                    spec.get('expected_sha256', entry['sha256']) == entry['sha256'], 'US source pin mismatch')
            # Verify the full compressed resource as bytes without retaining it.
            with archive.extractfile(member) as stream:
                digest = sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == entry['sha256'], f'US resource digest mismatch: {rid}')
        for key in ('usaleep', 'acs'):
            require(acquired[key + '_dictionary']['sha256'] == mapping[key]['dictionary_sha256'],
                    'adapter dictionary binding mismatch')
        require(acquired['acs_technical']['sha256'] == mapping['acs']['technical_sha256'],
                'ACS technical document binding mismatch')
        path = scratch / 'US_A.CSV'
        with archive.extractfile(members[declared['usaleep_data']['destination']]) as stream, path.open('xb') as sink:
            shutil.copyfileobj(stream, sink)
        us_source = _source_file(path, mapping['usaleep'], 'usaleep_2010_2015',
                                 declared['usaleep_data']['url'], acquired['usaleep_terms']['sha256'])
        _, us_audit = load_usaleep([us_source], mapping, numeric_identifiers=True)
        metadata = us_audit['metadata'][['original_id', 'state_fips', 'county_fips',
                                         'mortality_input_flag', 'primary_label_available']].copy()
        metadata = metadata[metadata.state_fips.isin(states.values())].reset_index(drop=True)
        del us_audit
        # ZIP central directory is small enough to hold this 25 MB resource.
        with archive.extractfile(members[declared['acs_geography']['destination']]) as stream:
            geography_bytes = stream.read()
        with zipfile.ZipFile(io.BytesIO(geography_bytes)) as geo:
            for state in states:
                wanted = f'g20105{state}.txt'
                names = [n for n in geo.namelist() if Path(n).name == wanted]
                require(len(names) == 1, f'missing or duplicate ACS geography: {state}')
                info = geo.getinfo(names[0])
                require(info.file_size <= 512 * 1024 * 1024, 'ACS geography member too large')
                with geo.open(info) as stream, (scratch / wanted).open('xb') as sink:
                    shutil.copyfileobj(stream, sink)
        del geography_bytes
        sequences = {t['sequence'] for t in mapping['acs']['tables'].values()}
        wanted = {f'{kind}20105{state}{seq}000.txt'
                  for state in states for seq in sequences for kind in ('e', 'm')}
        seen = set()
        with archive.extractfile(members[declared['acs_tracts']['destination']]) as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as inner:
                for member in inner:
                    name = Path(member.name).name
                    if name not in wanted:
                        continue
                    require(member.isfile() and name not in seen, 'duplicate or nonregular ACS sequence')
                    require(member.size <= 512 * 1024 * 1024, 'ACS sequence member too large')
                    seen.add(name)
                    with inner.extractfile(member) as chunk, (scratch / name).open('xb') as sink:
                        shutil.copyfileobj(chunk, sink)
        require(seen == wanted, 'missing approved ACS sequences')
    frames, manifests, missing = [], [us_source.manifest], 0
    registry = None
    for state in sorted(states):
        names = [f'g20105{state}.txt'] + sorted(n for n in wanted if n[6:8] == state)
        bundle = [_source_file(scratch / name, mapping['acs'], 'acs_2006_2010_5yr:' + name,
                  declared['acs_geography' if name.startswith('g') else 'acs_tracts']['url'] + '#' + name,
                  acquired['acs_terms']['sha256']) for name in names]
        frame, audit = load_acs(bundle, mapping)
        require(frame.original_id.str.startswith(states[state]).all(), 'ACS state identity mismatch')
        require(registry is None or registry == audit['registry'], 'ACS registry changes across states')
        registry = audit['registry']
        frames.append(frame)
        manifests.extend(audit['source_manifests'])
        missing += len(audit['missing_concepts'])
        del audit
        for item in bundle:
            item.path.unlink()
    full = pd.concat(frames, ignore_index=True).sort_values('original_id').reset_index(drop=True)
    require(not full.original_id.duplicated().any(), 'duplicate ACS tracts across states')
    require(set(metadata.original_id) <= set(full.original_id), 'USALEEP tract absent from ACS frame')
    audit = {'acs_tracts': len(full), 'usaleep_tracts': len(metadata),
             'flag_counts': metadata.mortality_input_flag.value_counts().sort_index().to_dict(),
             'acs_without_usaleep': sorted(set(full.original_id) - set(metadata.original_id)),
             'missing_concept_values': missing, 'geography_vintage': 2010,
             'acs_payload_sha256': acquired['acs_tracts']['sha256'],
             'mapping_sha256': mapping_hash(mapping)}
    return full, metadata, tuple(manifests), registry, audit


def typed_covariates(frame, metadata, sources, predictor_registry, request):
    """Bind the endpoint frame without publishing any Y or outcome metadata in X."""
    ids = tuple(sorted(metadata.original_id))
    require(bool(ids) and len(ids) == len(set(ids)), 'empty or duplicate endpoint IDs')
    graph = EntityGraph(original_ids=ids, links=())  # Exactly one observation per 2010 tract.
    special = [('original_id', 'identifier', ('linkage',)),
               ('life_expectancy_years', 'outcome', ('score',)),
               ('oxygen_deficit_mmhg', 'exposure', ('score',)),
               ('county_fips', 'county', ('county_routing',))]
    rules = tuple(FeatureRule(name=n, role=r, endpoints=(ENDPOINT,), uses=u,
                  approval_id='configs/approvals.yaml#plan_fixed.' +
                  ('county_membership' if r == 'county' else 'primary_outcome_flags')) for n, r, u in special)
    registry = FeatureRegistry(registry_id='tract-inputs-acs-2006-2010', rules=predictor_registry.rules + rules)
    spec = EstimandSpec(endpoint=ENDPOINT, target_id='usaleep-input-frame-before-support',
        outcome_scale='years', policy_id='unresolved-design-policy', weight_id='equal-tract',
        inference_unit='county', adjustment_schema_hash=registry.content_hash,
        source_lineage_hash=source_lineage_hash(sources))
    lineage = ArtifactLineage(source_hashes=tuple(s.payload_hash for s in sources), unit_ids=ids,
        parent_hashes=request.dependency_hashes, split_hash=None, config_hash=request.config_hash,
        model_hash=None, environment=(('code_identity', request.code_identity),), seed=None, parameter_count=None)
    columns = tuple(r.name for r in predictor_registry.rules)
    schema = (ColumnSpec(name='original_id', dtype='string'),
              ColumnSpec(name='life_expectancy_years', dtype='number', nullable=True),
              ColumnSpec(name='oxygen_deficit_mmhg', dtype='number', nullable=True),
              ColumnSpec(name='county_fips', dtype='string')) + tuple(
                  ColumnSpec(name=n, dtype='number', nullable=True) for n in columns)
    manifest = DataManifest(spec=spec, sources=sources, schema=schema, registry=registry, original_ids=ids,
        id_field='original_id', outcome_field='life_expectancy_years', exposure_field='oxygen_deficit_mmhg',
        weight_field=None, entity_graph_hash=graph.content_hash, lineage=lineage)
    selected = frame.set_index('original_id').loc[list(ids)]
    records = []
    for oid, row in selected.iterrows():
        record = {'original_id': oid, 'life_expectancy_years': None,
                  'oxygen_deficit_mmhg': None, 'county_fips': oid[:5]}
        record.update({n: None if pd.isna(row[n]) else float(row[n]) for n in columns})
        records.append(record)
    loaded = load_records(records, manifest, spec, manifest.schema_hash)
    return manifest, loaded.covariates(columns), graph
