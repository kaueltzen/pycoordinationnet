from copy import copy
from enum import Enum

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries

## -----------------------------------------------------------------------------

class AngleTypes(Enum):
    isolated = 0
    corner   = 1
    edge     = 2
    face     = 3

## -----------------------------------------------------------------------------

geometryDecoder = { i: geometry.mp_symbol for i, geometry in enumerate(AllCoordinationGeometries().get_geometries()) }
geometryEncoder = { geometry.mp_symbol: i for i, geometry in enumerate(AllCoordinationGeometries().get_geometries()) }

## -----------------------------------------------------------------------------

NumElements = len(Element)

# https://en.wikipedia.org/wiki/Oxidation_state:
# The highest known oxidation state is reported to be +9 in the tetroxoiridium(IX) cation (IrO+4)
# The lowest oxidation state is −5, as for boron in Al3BC.
NumOxidations = 15
NumAngleTypes = len(AngleTypes)
NumGeometries = len(geometryEncoder)

## -----------------------------------------------------------------------------

def encode_oxidation(i : int) -> int:
    if i < -5:
        raise ValueError(f'Unexpected oxidation state: {i}')
    if i >  9:
        raise ValueError(f'Unexpected oxidation state: {i}')
    return i + 5

def decode_oxidation(i : int) -> int:
    return i - 5

## -----------------------------------------------------------------------------

def encode_element(elem : str) -> int:
    return Element(elem).number

def decode_element(z : int) -> str:
    return Element.from_Z(z).value

## -----------------------------------------------------------------------------

def encode_ion(ion : str) -> int:
    if ion == 'cation':
        return 0
    if ion == 'anion':
        return 1
    raise ValueError(f'Unexpected ion string: {ion}')

def decode_ion(ion : int) -> str:
    return ['cation', 'anion'][ion]

## -----------------------------------------------------------------------------

def encode_ce_symbol(sym : str) -> int:
    return geometryEncoder[sym]

def decode_ce_symbol(sym : str) -> int:
    return geometryDecoder[sym]

## -----------------------------------------------------------------------------

def encode_sites(sites : 'Sites', array_type = list) -> 'Sites':
    sites = copy(sites)
    sites.oxidations = array_type(map(encode_oxidation, sites.oxidations))
    sites.elements   = array_type(map(encode_element  , sites.elements))
    sites.ions       = array_type(map(encode_ion      , sites.ions))
    return sites

## -----------------------------------------------------------------------------

def decode_sites(sites : 'Sites', array_type = list) -> 'Sites':
    sites = copy(sites)
    sites.oxidations = array_type(map(decode_oxidation, sites.oxidations))
    sites.elements   = array_type(map(decode_element  , sites.elements))
    sites.ions       = array_type(map(decode_ion      , sites.ions))
    return sites

## -----------------------------------------------------------------------------

def encode_ces(ces : list) -> list:
    ces = copy(ces)
    for i, _ in enumerate(ces):
        ces[i] = copy(ces[i])
        ces[i]['ce_symbol'] = encode_ce_symbol(ces[i]['ce_symbol'])
    return ces

## -----------------------------------------------------------------------------

def decode_ces(ces : list) -> list:
    ces = copy(ces)
    for i, _ in enumerate(ces):
        ces[i] = copy(ces[i])
        ces[i]['ce_symbol'] = decode_ce_symbol(ces[i]['ce_symbol'])
    return ces

## -----------------------------------------------------------------------------

def encode_neighbors(ce_neighbors : list) -> list:
    ce_neighbors = copy(ce_neighbors)
    for i, _ in enumerate(ce_neighbors):
        ce_neighbors[i] = copy(ce_neighbors[i])
        ce_neighbors[i]['connectivity'] = AngleTypes[ce_neighbors[i]['connectivity']].value
    return ce_neighbors

## -----------------------------------------------------------------------------

def decode_neighbors(ce_neighbors : list) -> list:
    ce_neighbors = copy(ce_neighbors)
    for i, _ in enumerate(ce_neighbors):
        ce_neighbors[i] = copy(ce_neighbors[i])
        ce_neighbors[i]['connectivity'] = AngleTypes(ce_neighbors[i]['connectivity']).name
    return ce_neighbors

## -----------------------------------------------------------------------------

def encode_features(features : 'Crytures') -> 'Crytures':
    features              = copy(features)
    features.sites        = encode_sites    (features.sites)
    features.ces          = encode_ces      (features.ces)
    features.ce_neighbors = encode_neighbors(features.ce_neighbors)
    return features

def decode_features(features : 'Crytures') -> 'Crytures':
    features              = copy(features)
    features.sites        = decode_sites    (features.sites)
    features.ces          = decode_ces      (features.ces)
    features.ce_neighbors = decode_neighbors(features.ce_neighbors)
    return features
