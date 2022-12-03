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

def encode_features(features : dict, array_type = list) -> dict:
    features      = copy(features)
    features.base = copy(features.base)
    features.ces  = copy(features.ces)
    # Convert base
    features.base.oxidations = array_type(map(encode_oxidation, features.base.oxidations))
    features.base.elements   = array_type(map(encode_element  , features.base.elements))
    features.base.ions       = array_type(map(encode_ion      , features.base.ions))
    # Convert coordination environments
    features.ces.ce_symbols  = array_type(map(encode_ce_symbol, features.ces.ce_symbols))
    return features

def decode_features(features : dict, array_type = list) -> dict:
    features      = copy(features)
    features.base = copy(features.base)
    features.ces  = copy(features.ces)
    # Convert base
    features.base.oxidations = array_type(map(decode_oxidation, features.base.oxidations))
    features.base.elements   = array_type(map(decode_element  , features.base.elements))
    features.base.ions       = array_type(map(decode_ion      , features.base.ions))
    # Convert coordination environments
    features.ces.ce_symbols  = array_type(map(decode_ce_symbol, features.ces.ce_symbols))
    return features
