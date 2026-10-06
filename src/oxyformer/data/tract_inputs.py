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


def _check_resource_pins(spec, size, digest):
    # The acquisition schema represents absent optional pins either by omission
    # or by null. Recorded receipt hashes are still always verified separately.
    for key, actual in [('expected_bytes', size), ('expected_sha256', digest)]:
        require(spec.get(key) is None or spec[key] == actual, 'source pin mismatch: ' + key)


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
            _check_resource_pins(spec, member.size, entry['sha256'])
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
            with tarfile.open(fileobj=stream, mode='r|gz', bufsize=1024 * 1024) as inner:
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


TASK_FILE = 'configs/execution/tasks/tract.yaml'
APPROVAL_REFERENCE = 'configs/approvals.yaml#owner_decisions.tract_design'


def tract_decisions():
    """Read owner values; a changed decision needs a newly reviewed mapping."""
    from oxyformer.execution.runner import read_mapping
    approved = read_mapping(ROOT / 'configs/approvals.yaml')
    decision = approved.get('owner_decisions', {}).get('tract_design')
    task_document = read_mapping(ROOT / TASK_FILE)
    require(isinstance(decision, dict) and sha256(canonical_json(decision).encode()).hexdigest()
            == task_document['tract_design_sha256'], 'tract design owner decision absent or changed')
    return decision


def dispatch_inputs(request):
    """Validate the committed task/envelope and resolve named, hash-bound files."""
    from oxyformer.execution.runner import read_mapping
    request.verify_inputs()
    envelope = read_mapping(request.config_path)
    task = json.loads(Path(request.task_path).read_text())
    expected = next((t for t in read_mapping(ROOT / TASK_FILE)['tasks'] if t['stage'] == request.stage), None)
    require(task == expected and envelope['stage'] == request.stage, 'tract task differs from repository')
    approvals = ROOT / 'configs/approvals.yaml'
    require(envelope['approvals'] == read_mapping(approvals)
            and envelope['input_sources'].get(str(approvals)) == file_hash(approvals),
            'dispatcher approvals differ from repository owner file')
    tract_decisions()
    hashes = dict(zip(request.dependency_paths, request.dependency_hashes))
    paths = {}
    for unit, names in task['needs'].items():
        root = Path(envelope['dependencies'][unit])
        for name in names:
            path = root / name
            require(str(path) in hashes, 'tract dependency is not hash-bound')
            paths[(unit, name)] = path
    return paths


def sf1_coordinates(archive, state, fips):
    """Population-weighted mean of SF1 block internal-point latitudes/longitudes.

    2010 SF1 dictionary 6-8 and endnotes 21/22: INTPTLAT at 337 (11
    characters), INTPTLON at 348 (12), explicit signed decimal degrees.
    https://www2.census.gov/programs-surveys/decennial/2010/technical-documentation/complete-tech-docs/summary-file/sf1.pdf
    Only SUMLEV 101 / GEOCOMP 00 / CHARITER 000 participates. The merged
    parser reconciles block, tract and state population with segment 01.
    """
    import math
    import re
    from oxyformer.exposure.census_blocks import read_sf1_population
    pop = read_sf1_population(archive, state, fips).set_index('block_id')
    points = {}
    with zipfile.ZipFile(archive) as z, z.open(state.lower() + 'geo2010.sf1') as stream:
        for raw in stream:
            if raw[8:16] != b'10100000':
                continue
            bid = (raw[27:32] + raw[54:60] + raw[61:65]).decode('ascii')
            require(bid not in points and bid in pop.index, 'duplicate or unmatched SF1 point')
            lat, lon = raw[336:347].decode('ascii'), raw[347:359].decode('ascii')
            require(re.fullmatch(r'[+-][0-9]{2}\.[0-9]{7}', lat) is not None and
                    re.fullmatch(r'[+-][0-9]{3}\.[0-9]{7}', lon) is not None, 'invalid SF1 internal point')
            lat, lon = float(lat), float(lon)
            require(-90 <= lat <= 90 and -180 <= lon <= 180, 'SF1 point outside geographic range')
            points[bid] = (lat, lon)
    require(set(points) == set(pop.index), 'SF1 internal point omissions')
    pop['latitude'] = [points[b][0] for b in pop.index]
    pop['longitude'] = [points[b][1] for b in pop.index]
    result = {}
    for tract, group in pop.groupby('tract_id', sort=True):
        total = sum(int(p) for p in group.population)
        # A zero-population tract has no population-weighted coordinate. Do not
        # replace it with an unweighted point. Endpoint rows require a point.
        if total:
            result[tract] = (math.fsum(p * lat for p, lat in zip(group.population, group.latitude)) / total,
                             math.fsum(p * lon for p, lon in zip(group.population, group.longitude)) / total)
    return result


def census_coordinates(payload, receipt_path, scratch, *, source=None, states=None):
    from oxyformer.exposure.archives import extract_member
    source = load_source('census') if source is None else source
    states = yaml.safe_load((ROOT / 'configs/exposure.yaml').read_text())['state_fips'] if states is None else states
    receipt = json.loads(read_regular(receipt_path))
    require(receipt['status'] == 'complete' and receipt['manifest_id'] == 'census' and
            receipt['manifest_sha256'] == sha256(canonical_json(source).encode()).hexdigest(),
            'Census receipt differs from repository source manifest')
    declared = {r['id']: r for r in source['resources']}
    acquired = {r['id']: r for r in receipt['resources']}
    require(len(acquired) == len(receipt['resources']) and acquired.keys() == declared.keys(),
            'Census resource inventory mismatch')
    verify_input_hash(payload, receipt['payload_sha256'])
    require(Path(payload).stat().st_size == receipt['payload_bytes'], 'Census payload size mismatch')
    result = {}
    for state, fips in sorted(states.items()):
        rid = 'sf1_' + state.lower()
        entry, spec = acquired[rid], declared[rid]
        require(entry['destination'] == spec['destination'] and entry['bytes'] <= spec['max_bytes'],
                'Census resource identity mismatch')
        _check_resource_pins(spec, entry['bytes'], entry['sha256'])
        path = Path(scratch) / (rid + '.zip')
        extract_member(payload, entry['destination'], path, entry['sha256'])
        current = sf1_coordinates(path, state, fips)
        require(not result.keys() & current.keys(), 'overlapping Census tracts')
        result.update(current)
        path.unlink()
    return result


def typed_geography(manifest, metadata, coordinates):
    from math import floor
    from pyproj import Transformer
    from oxyformer.design.eligibility import GeographyRow, GeographyTable
    decision = tract_decisions()
    grid = decision['subblock']
    # Census 2010 internal points use the North American Datum of 1983.
    project = Transformer.from_crs('EPSG:4269', grid['crs'], always_xy=True)
    records = metadata.set_index('original_id')
    rows = []
    for oid in manifest.original_ids:
        require(oid in coordinates, 'endpoint tract lacks a population-weighted Census coordinate: ' + oid)
        lat, lon = coordinates[oid]
        x, y = project.transform(lon, lat, errcheck=True)
        i, j = (floor((v - o) / grid['cell_size_m']) for v, o in zip((x, y), grid['origin_m']))
        record = records.loc[oid]
        rows.append(GeographyRow(original_id=oid, tract_id=oid, county=oid[:5], state=oid[:2],
            subblock=f'{oid[:5]}:{i}:{j}', assignment_geography=oid, latitude=lat, longitude=lon,
            outcome_flag=int(record.mortality_input_flag), label_available=bool(record.primary_label_available)))
    return GeographyTable(rows=tuple(rows), data_manifest_hash=manifest.content_hash, county_field='county_fips',
        approval_reference='configs/approvals.yaml#plan_fixed.county_membership',
        mapping_review_id=APPROVAL_REFERENCE)


def run_stage(request):
    """Publish only outcome-free typed inputs and the full approved ACS pool."""
    import tempfile
    from dataclasses import replace
    from oxyformer.contracts import StageResult
    from oxyformer.execution.paths import atomic_write, atomic_json
    from oxyformer.provenance import ArtifactRecord
    require(request.stage == 'tract-inputs', 'unexpected tract input stage')
    paths = dispatch_inputs(request)
    out = Path(request.output_dir)
    with tempfile.TemporaryDirectory(prefix='.tract-', dir=out) as scratch:
        frame, metadata, sources, registry, audit = load_us_inputs(
            paths['fetch-us', 'payload.tar'], paths['fetch-us', 'receipts.json'], scratch)
        coordinates = census_coordinates(paths['fetch-census', 'payload.tar'],
                                          paths['fetch-census', 'receipts.json'], scratch)
        manifest, covariates, graph = typed_covariates(frame, metadata, sources, registry, request)
        geography = typed_geography(manifest, metadata, coordinates)
        for name, value in [('data_manifest', manifest), ('covariates', covariates),
                            ('entity_graph', graph), ('geography', geography)]:
            atomic_write(out, name + '.json', value.to_json())
        # Full outcome-independent pool; no coordinates, flags, Y, SE or fitted transforms.
        frame.to_parquet(Path(scratch) / 'acs_pool.parquet', index=False)
        import os
        os.link(Path(scratch) / 'acs_pool.parquet', out / 'acs_pool.parquet')
        audit.update(census_tract_coordinates=len(coordinates), tract_design=tract_decisions(),
                     census_payload_sha256=file_hash(paths['fetch-census', 'payload.tar']))
        atomic_json(out, 'input_audit.json', audit)
    task = json.loads(Path(request.task_path).read_text())
    pool_lineage = replace(manifest.lineage, unit_ids=tuple(frame.original_id))
    return StageResult(request_hash=request.content_hash, status='pass', message='Typed outcome-free tract inputs built',
        artifacts=tuple(ArtifactRecord(path=name, sha256=file_hash(out / name),
                        lineage=pool_lineage if name == 'acs_pool.parquet' else manifest.lineage,
                        kind='tract_input') for name in task['outputs']))
