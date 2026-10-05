"""2010 TIGER2010BLKPOPHU + SF1 state ZIP reader, with exact block joins.

Inventory: configs/sources/census.json via load_source('census'). Layouts inspected:
https://www2.census.gov/programs-surveys/decennial/2010/technical-documentation/complete-tech-docs/summary-file/sf1.pdf
(chapters 4/6: SUMLEV 101, component 00, iteration 000; segment 01 P0010001,
called P001001 by the API; fixed-width geography and LOGRECNO linking).
https://meta.geo.census.gov/data/existing/decennial/GEO/GPMB/Tabblock/Tabblock2010/2010_pophu.ea.iso.xml
(BLOCKID10, STATEFP10, COUNTYFP10, TRACTCE10, POP10).
Only geography and segment 01 are read from SF1; GDAL reads the block ZIP in
place. No archive extraction, downloads, outcomes or learned weights.
"""
import csv
from decimal import Decimal
from numbers import Real
import io
from pathlib import Path
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from oxyformer.data.source_manifest import load_source
from oxyformer.provenance import require

# Zero-based, half-open slices from the 2010 SF1 state geographic header.
GEO_FIELDS = {'state': (27, 29), 'county': (29, 32), 'tract': (54, 60),
              'block': (61, 65), 'logrecno': (18, 25), 'population': (318, 327)}


def _is_population_count(value):
    # pandas object/nullable storage must not change the numeric admission rule.
    # No string coercion, truth values, complex numbers or tensor inputs.
    if not isinstance(value, (Real, Decimal)) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        integer = int(value)
    except (ValueError, OverflowError):
        return False
    return integer >= 0 and value == integer


def validate_blocks(blocks):
    require(isinstance(blocks, gpd.GeoDataFrame) and blocks.crs is not None,
            'block geography requires CRS')
    require(set(blocks.columns) == {'block_id', 'tract_id', 'population', 'geometry'},
            'only block identifiers, fixed population and geometry may enter exposure')
    require(len(blocks) > 0 and not blocks.block_id.duplicated().any(), 'empty or duplicate blocks')
    require(blocks.block_id.map(lambda x: isinstance(x, str) and len(x) == 15 and
            x.isascii() and x.isdigit()).all(), '2010 block IDs must be 15-digit strings')
    require((blocks.block_id.str[:11] == blocks.tract_id).all(), 'tract membership mismatch')
    require(all(_is_population_count(value) for value in blocks.population),
            'invalid Census population')
    require(blocks.geometry.notna().all() and (~blocks.geometry.is_empty).all() and
            blocks.geometry.is_valid.all() and blocks.geom_type.isin(['Polygon', 'MultiPolygon']).all(),
            'invalid block polygons')
    return blocks.sort_values('block_id').reset_index(drop=True)


def read_sf1_population(archive, state_abbreviation, state_fips):
    """Return all blocks and reconcile segment totals with geographic POP100.

    The 040 state total and 140 tract totals also have to equal the block sums.
    Other hierarchy/characteristic records are deliberately not population rows.
    """
    prefix = state_abbreviation.lower()
    geography, totals = {}, {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        geo_name, data_name = prefix + 'geo2010.sf1', prefix + '000012010.sf1'
        require(names.count(geo_name) == names.count(data_name) == 1, 'required SF1 members missing/duplicate')
        with z.open(geo_name) as stream:
            for raw in stream:
                # One byte per fixed-width position; NAME is not interpreted.
                # Decode only selected codes below as ASCII, not unused names.
                line = raw.decode('latin-1')
                require(len(line.rstrip('\r\n')) == 500, 'SF1 geography record must be 500 characters')
                if line[8:11] not in ('101', '140', '040') or line[11:13] != '00' or line[13:16] != '000':
                    continue
                require(line[:6].strip() == 'SF1ST' and line[6:8] == state_abbreviation and
                        line[27:29] == state_fips, 'SF1 state/file mismatch')
                values = {k: raw[a:b].decode('ascii') for k, (a, b) in GEO_FIELDS.items()}
                key = values['logrecno']
                require(key.isdigit() and key not in geography, 'duplicate/invalid SF1 LOGRECNO')
                level = line[8:11]
                ident = state_fips + (values['county'] + values['tract'] if level != '040' else '')
                if level == '101':
                    ident += values['block']
                require(ident.isdigit(), 'invalid SF1 geographic identifier')
                geography[key] = (level, ident, int(values['population']))
        with z.open(data_name) as stream:
            for row in csv.reader(io.TextIOWrapper(stream, encoding='ascii')):
                # Official 2010 SF1 p. 6-21: file 01 has P1 only. P2 is
                # in file 02, P3 in file 03; this is not the 2000 SF1 layout.
                require(len(row) == 6, 'SF1 segment 01 requires five linking fields and P0010001')
                require(row[:4] == ['SF1ST', state_abbreviation, '000', '01'], 'SF1 segment identity mismatch')
                key = row[4]
                if key in geography:
                    require(key not in totals and row[5].isdigit(), 'duplicate/invalid SF1 population')
                    totals[key] = int(row[5])
        require(set(totals) == set(geography), 'missing SF1 population rows')
    records, tract_totals, state_totals = [], {}, []
    for key, (level, ident, header_pop) in geography.items():
        pop = totals[key]
        require(pop == header_pop, 'P001001 disagrees with POP100')
        if level == '101':
            records.append((ident, ident[:11], pop))
        elif level == '140':
            require(ident not in tract_totals, 'duplicate SF1 tract')
            tract_totals[ident] = pop
        else:
            state_totals.append(pop)
    result = pd.DataFrame(records, columns=['block_id', 'tract_id', 'population'])
    require(len(result) > 0 and not result.block_id.duplicated().any(), 'empty/duplicate SF1 blocks')
    require(result.groupby('tract_id').population.sum().to_dict() == tract_totals,
            'SF1 block/tract population mismatch')
    require(state_totals == [int(result.population.sum())], 'SF1 block/state population mismatch')
    return result


def read_census_blocks(block_archive, sf1_archive, *, state_abbreviation, state_fips,
                       source_config=None):
    """Read verified local archive paths selected from the committed inventory.

    The caller verifies archive bytes against acquisition receipts before entry.
    source_config is injectable only for synthetic fixtures.
    """
    config = load_source('census') if source_config is None else source_config
    require(config['release_identity']['releases']['geography_vintage'] == 2010,
            'Census geography vintage must be 2010')
    inventory = {r['id']: r for r in config['resources']}
    require('blocks_' + state_fips in inventory and 'sf1_' + state_abbreviation.lower() in inventory,
            'state archives absent from reviewed Census inventory')
    stem = Path(inventory['blocks_' + state_fips]['url']).name.removesuffix('.zip')
    with zipfile.ZipFile(block_archive) as archive:
        require(all(archive.namelist().count(stem + suffix) == 1
                    for suffix in ('.shp', '.shx', '.dbf', '.prj')), 'required block ZIP members missing')
    geo = gpd.read_file(f'zip://{Path(block_archive).resolve()}!{stem}.shp')
    columns = {'BLOCKID10', 'STATEFP10', 'COUNTYFP10', 'TRACTCE10', 'POP10'}
    require(columns <= set(geo.columns), '2010 block attribute layout mismatch')
    require((geo.STATEFP10 == state_fips).all(), 'block state mismatch')
    require((geo.BLOCKID10.str[:11] == geo.STATEFP10 + geo.COUNTYFP10 + geo.TRACTCE10).all(),
            'block geographic identifier mismatch')
    require(not geo.BLOCKID10.duplicated().any(), 'duplicate block geometry; unresolved block parts')
    pop = read_sf1_population(sf1_archive, state_abbreviation, state_fips)
    require(set(geo.BLOCKID10) == set(pop.block_id), 'unmatched Census blocks; no silent omissions')
    geo = geo.merge(pop, left_on='BLOCKID10', right_on='block_id', validate='one_to_one')
    require((geo.POP10 == geo.population).all(), 'TIGER/SF1 population mismatch')
    return validate_blocks(geo[['block_id', 'tract_id', 'population', 'geometry']])
