import numpy as np
import os
import pytest

from pymatgen.core import Structure
from monty.serialization import loadfn

from crytures import crysFeaturizer, mp_icsd_clean
from crytures.featurizer import analyze_env
from crytures.utility    import oxide_check

## -----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## -----------------------------------------------------------------------------

@pytest.fixture
def testDataFix():
    testData = loadfn(os.path.join(root, 'test_data.json.gz'))
    for Tdatum in testData:
        Tdatum['structure'] = Structure.from_dict(Tdatum['structure'])
    return testData

## -----------------------------------------------------------------------------

def test_exper_data_cleaning(testDataFix):
    _, baddata = mp_icsd_clean(testDataFix, reportBadData = True)

    assert all(item in baddata['other_anion_IDs'] for item in ('mp-5634', 'mp-788'))
    assert len(baddata['other_anion_IDs']) == 2
    assert 'mp-5634' not in baddata['other_oxidation_IDs']
    assert any(item['material_id'] not in baddata['valence_problem_IDs'] for item in testDataFix)
    assert any(item['material_id'] not in baddata['bad_structure_IDs']   for item in testDataFix)

## -----------------------------------------------------------------------------

@pytest.fixture
def actual_oxide_check():
    return loadfn(os.path.join(root, 'test_oxide_check.json.gz'))

## -----------------------------------------------------------------------------

def test_oxide_check(actual_oxide_check, testDataFix):    
    for Tdatum in testDataFix:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
        print(type(Tdatum['material_id']))
        print(Tdatum['material_id'])
        assert other_anion == actual_oxide_check[Tdatum['material_id']]['other_anion']
        assert other_oxidation == actual_oxide_check[Tdatum['material_id']]['other_oxidation']
        assert bad_structure == actual_oxide_check[Tdatum['material_id']]['bad_structure']
        # TODO: There could be permutations or machine percision difference
        assert np.isclose(primStruc.volume , actual_oxide_check[Tdatum['material_id']]['primStruc'].volume)
        i = Tdatum['material_id']
        print(f'Oxide_check function tests were passed for material id {i}')

## -----------------------------------------------------------------------------

@pytest.fixture
def actual_analyze_env():
    return loadfn(os.path.join(root, 'test_env.json.gz'))

def test_analyze_env(actual_analyze_env, testDataFix):    
    for Tdatum in testDataFix:
        oxid_states, sc = analyze_env(Tdatum['structure'], mystrategy='simple')
        assert oxid_states == actual_analyze_env[Tdatum['material_id']]['oxid_states']
        assert sc.as_dict()['connectivity_graph'] == actual_analyze_env[Tdatum['material_id']]['sc'].as_dict()['connectivity_graph']

## -----------------------------------------------------------------------------

@pytest.fixture
def actual_crysFeaturizer():
    data = loadfn(os.path.join(root, 'test_features.json.gz'))
    for matID in data.keys():
        data[matID] = { int(k): v for k, v in data[matID].items() }
    return data

## -----------------------------------------------------------------------------

def test_firstDegreeFeatures(actual_crysFeaturizer, actual_analyze_env):
    for matID in actual_crysFeaturizer.keys():
        featureTest_dict = crysFeaturizer(
            SC_object      = actual_analyze_env[matID]['sc'],
            oxidation_list = actual_analyze_env[matID]['oxid_states'])

        for atomIndex in actual_crysFeaturizer[matID].keys(): 
            assert(actual_crysFeaturizer[matID][atomIndex]['oxidation'] == featureTest_dict[atomIndex]['oxidation'])
            assert(actual_crysFeaturizer[matID][atomIndex]['element'  ] == featureTest_dict[atomIndex]['element'])
            np.testing.assert_allclose(actual_crysFeaturizer[matID][atomIndex]['coords'], featureTest_dict[atomIndex]['coords'])
            assert(actual_crysFeaturizer[matID][atomIndex]['ion'] == featureTest_dict[atomIndex]['ion'])

            if actual_crysFeaturizer[matID][atomIndex]['ion'] == 'cation':
                assert(actual_crysFeaturizer[matID][atomIndex]['localEnv'] == featureTest_dict[atomIndex]['localEnv'])
                for d, NN in enumerate(actual_crysFeaturizer[matID][atomIndex]['NN_distances']):
                    assert(pytest.approx(NN[0],0.001) == featureTest_dict[atomIndex]['NN_distances'][d][0]) #Nneighbor distance
                    assert(NN[1] == featureTest_dict[atomIndex]['NN_distances'][d][1]) #neigbor element

## -----------------------------------------------------------------------------

def test_nnnFeatures(actual_crysFeaturizer, actual_analyze_env):    
    for matID in actual_crysFeaturizer.keys():

        featureTest_dict = crysFeaturizer(
            SC_object      = actual_analyze_env[matID]['sc'],
            oxidation_list = actual_analyze_env[matID]['oxid_states'])

        for atomIndex in actual_crysFeaturizer[matID].keys(): 

            if actual_crysFeaturizer[matID][atomIndex]['ion'] == 'cation':
                for p, NNN in enumerate(actual_crysFeaturizer[matID][atomIndex]['poly_distances']):
                    assert(pytest.approx(NNN[0],0.001) == featureTest_dict[atomIndex]['poly_distances'][p][0]) #NNN distance
                    assert(NNN[1] == featureTest_dict[atomIndex]['poly_distances'][p][1]) #NNN element

                for c, connectivity in enumerate(actual_crysFeaturizer[matID][atomIndex]['connectivity_angles']):
                    assert(connectivity[0]==featureTest_dict[atomIndex]['connectivity_angles'][c][0]) #checks connectivity type (cornder/edge/face/noConnection)
                    for connectivityIndex in range(1, len(connectivity)):
                        assert(pytest.approx(connectivity[connectivityIndex][0], 0.001)==featureTest_dict[atomIndex]['connectivity_angles'][c][connectivityIndex][0])#angle
                        assert(connectivity[connectivityIndex][1]==featureTest_dict[atomIndex]['connectivity_angles'][c][connectivityIndex][1])#element
