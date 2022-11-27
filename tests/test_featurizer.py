import numpy as np
import os
import pytest

from pymatgen.core import Structure
from monty.serialization import loadfn

from crytures            import featurize, mp_icsd_clean
from crytures.featurizer import analyze_environment, compute_features_first_degree
from crytures.utility    import oxide_check

## -----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## -----------------------------------------------------------------------------

@pytest.fixture
def testData():
    testData = loadfn(os.path.join(root, 'test_data.json.gz'))
    for Tdatum in testData:
        Tdatum['structure'] = Structure.from_dict(Tdatum['structure'])
    return testData

## -----------------------------------------------------------------------------

def test_exper_data_cleaning(testData):
    _, baddata = mp_icsd_clean(testData, reportBadData = True)

    assert all(item in baddata['other_anion_IDs'] for item in ('mp-5634', 'mp-788'))
    assert len(baddata['other_anion_IDs']) == 2
    assert 'mp-5634' not in baddata['other_oxidation_IDs']
    assert any(item['material_id'] not in baddata['valence_problem_IDs'] for item in testData)
    assert any(item['material_id'] not in baddata['bad_structure_IDs']   for item in testData)

## -----------------------------------------------------------------------------

@pytest.fixture
def oxide_check_true():
    return loadfn(os.path.join(root, 'test_oxide_check.json.gz'))

## -----------------------------------------------------------------------------

def test_oxide_check(oxide_check_true, testData):    
    for Tdatum in testData:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
        assert other_anion     == oxide_check_true[Tdatum['material_id']]['other_anion']
        assert other_oxidation == oxide_check_true[Tdatum['material_id']]['other_oxidation']
        assert bad_structure   == oxide_check_true[Tdatum['material_id']]['bad_structure']
        # TODO: There could be permutations or machine percision difference
        assert np.isclose(primStruc.volume , oxide_check_true[Tdatum['material_id']]['primStruc'].volume)

## -----------------------------------------------------------------------------

@pytest.fixture
def env_true():
    return loadfn(os.path.join(root, 'test_env.json.gz'))

## -----------------------------------------------------------------------------

def test_analyze_env(env_true, testData):    
    for Tdatum in testData:
        sc, oxid_states = analyze_environment(Tdatum['structure'], mystrategy='simple')
        assert oxid_states == env_true[Tdatum['material_id']]['oxid_states']
        assert sc.as_dict()['connectivity_graph'] == env_true[Tdatum['material_id']]['sc'].as_dict()['connectivity_graph']

## -----------------------------------------------------------------------------

@pytest.fixture
def features_true_list():
    data = loadfn(os.path.join(root, 'test_features.json.gz'))
    for matID in data.keys():
        data[matID] = { int(k): v for k, v in data[matID].items() }
    return data

## -----------------------------------------------------------------------------

def test_firstDegreeFeatures(features_true_list, testData):
    for Tdatum in testData:
        features_true = features_true_list[Tdatum['material_id']]

        structure_connectivity, oxid_states = analyze_environment(Tdatum['structure'], mystrategy = 'simple')
        features_test = compute_features_first_degree(structure_connectivity, oxid_states)

        for atomIndex in features_true.keys(): 
            assert(features_true[atomIndex]['oxidation'] == features_test[atomIndex]['oxidation'])
            assert(features_true[atomIndex]['element'  ] == features_test[atomIndex]['element'])
            np.testing.assert_allclose(features_true[atomIndex]['coordinates'], features_test[atomIndex]['coordinates'])
            assert(features_true[atomIndex]['ion'] == features_test[atomIndex]['ion'])

            if features_true[atomIndex]['ion'] == 'cation':
                assert(features_true[atomIndex]['ce'] == features_test[atomIndex]['ce'])
                for k, neighbor in enumerate(features_true[atomIndex]['distances']):
                    # Test neighbor distance
                    assert(pytest.approx(neighbor[1], 0.001) == features_test[atomIndex]['distances'][k][1])
                    # Test neigbor element
                    assert(neighbor[0] == features_test[atomIndex]['distances'][k][0])

## -----------------------------------------------------------------------------

def test_nnnFeatures(features_true_list, testData):    
    for Tdatum in testData:
        features_true = features_true_list[Tdatum['material_id']]
        features_test = featurize(Tdatum['structure'])

        for atomIndex in features_true.keys():

            # Make sure the atom order is preserved
            assert (features_true[atomIndex]['coordinates'] == features_test[atomIndex]['coordinates']).all()

            if features_true[atomIndex]['ion'] == 'cation':
                # The order of distances may vary
                distances_true = []
                distances_test = []
                elements_true  = []
                elements_test  = []
                for p, nnn in enumerate(features_true[atomIndex]['ce_distances']):
                    # Extract NNN distance
                    distances_true.append(round(nnn[2], 3))
                    distances_test.append(round(features_test[atomIndex]['ce_distances'][p][2], 3))
                    # Extract NNN element
                    elements_true.append(nnn[1])
                    elements_test.append(features_test[atomIndex]['ce_distances'][p][1])

                assert(np.all(np.sort(distances_true) == np.sort(distances_test)))
                assert(np.all(np.sort( elements_true) == np.sort( elements_test)))

                types_true    = []
                types_test    = []
                angles_true   = []
                angles_test   = []
                elements_true = []
                elements_test = []
                for connectivity in features_true[atomIndex]['ce_angles']:
                    # Check connectivity type (cornder/edge/face/noConnection)
                    types_true.append(connectivity[0])

                    for connectivityIndex in range(1, len(connectivity)):
                        # Extract angles
                        angles_true.append(round(connectivity[connectivityIndex][3], 3))
                        # Extract elements
                        elements_true.append(connectivity[connectivityIndex][2])

                for connectivity in features_test[atomIndex]['ce_angles']:
                    # Check connectivity type (cornder/edge/face/noConnection)
                    types_test.append(connectivity[0])

                    for connectivityIndex in range(1, len(connectivity)):
                        # Extract angles
                        angles_test.append(round(connectivity[connectivityIndex][3], 3))
                        # Extract elements
                        elements_test.append(connectivity[connectivityIndex][2])

                assert(np.all(np.sort(   types_true) == np.sort(   types_test)))
                assert(np.all(np.sort(  angles_true) == np.sort(  angles_test)))
                assert(np.all(np.sort(elements_true) == np.sort(elements_test)))
