
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

def encode_distances(distances : list) -> list:
    distances = distances.copy()
    for i, distance in enumerate(distances):
        distances[i] = (
            encode_element(distance[0]),
            encode_element(distance[1]),
            distance[2])
    return distances

def decode_distances(distances : list) -> list:
    distances = distances.copy()
    for i, distance in enumerate(distances):
        distances[i] = (
            decode_element(distance[0]),
            decode_element(distance[1]),
            distance[2])
    return distances

## -----------------------------------------------------------------------------

def encode_ce(ces : list) -> list:
    ces = ces.copy()
    for i, ce in enumerate(ces):
        ce = ce.copy()
        ce['ce_symbol'] = geometryEncoder[ce['ce_symbol']]
        ces[i] = ce
    return ces

def decode_ce(ces : list) -> list:
    ces = ces.copy()
    for i, ce in enumerate(ces):
        ce = ce.copy()
        ce['ce_symbol'] = geometryDecoder[ce['ce_symbol']]
        ces[i] = ce
    return ces

## -----------------------------------------------------------------------------

def encode_angle(angle : tuple[4]) -> tuple[4]:
    return (
        encode_element(angle[0]),
        encode_element(angle[1]),
        encode_element(angle[2]),
        angle[3]
    )

def decode_angle(angle : tuple[4]) -> tuple[4]:
    return (
        decode_element(angle[0]),
        decode_element(angle[1]),
        decode_element(angle[2]),
        angle[3]
    )

## -----------------------------------------------------------------------------

def encode_ce_angles(ce_angles : list) -> list:
    ce_angles = ce_angles.copy()
    for i, ce_angle in enumerate(ce_angles):
        ce_angle    = ce_angle.copy()
        ce_angle[0] = AngleTypes[ce_angle[0]].value
        for j in range(1, len(ce_angle)):
            ce_angle[j] = encode_angle(ce_angle[j])
        ce_angles[i] = ce_angle
    return ce_angles

def decode_ce_angles(ce_angles : list) -> list:
    ce_angles = ce_angles.copy()
    for i, ce_angle in enumerate(ce_angles):
        ce_angle    = ce_angle.copy()
        ce_angle[0] = AngleTypes(ce_angle[0]).name
        for j in range(1, len(ce_angle)):
            ce_angle[j] = decode_angle(ce_angle[j])
        ce_angles[i] = ce_angle
    return ce_angles

## -----------------------------------------------------------------------------

def encode_site_features(features : dict) -> dict:
    features = features.copy()
    features['element'  ] = encode_element  (features['element'  ])
    features['oxidation'] = encode_oxidation(features['oxidation'])
    features['ion'      ] = encode_ion      (features['ion'      ])
    if 'distances' in features.keys():
        features['distances'] = encode_distances(features['distances'])
    if 'ce' in features.keys():
        features['ce'] = encode_ce(features['ce'])
    if 'ce_distances' in features.keys():
        features['ce_distances'] = encode_distances(features['ce_distances'])
    if 'ce_angles' in features.keys():
        features['ce_angles'] = encode_ce_angles(features['ce_angles'])
    return features

def decode_site_features(features : dict) -> dict:
    features = features.copy()
    features['element'  ] = decode_element  (features['element'  ])
    features['oxidation'] = decode_oxidation(features['oxidation'])
    features['ion'      ] = decode_ion      (features['ion'      ])
    if 'distances' in features.keys():
        features['distances'] = decode_distances(features['distances'])
    if 'ce' in features.keys():
        features['ce'] = decode_ce(features['ce'])
    if 'ce_distances' in features.keys():
        features['ce_distances'] = decode_distances(features['ce_distances'])
    if 'ce_angles' in features.keys():
        features['ce_angles'] = decode_ce_angles(features['ce_angles'])
    return features

## -----------------------------------------------------------------------------

def encode_features(features : dict) -> dict:
    features = features.copy()
    for i in features:
        features[i] = encode_site_features(features[i])
    return features

def decode_features(features : dict) -> dict:
    features = features.copy()
    for i in features:
        features[i] = decode_site_features(features[i])
    return features
