# This supresses warnings.
import warnings
warnings.filterwarnings('ignore')

## -----------------------------------------------------------------------------

import sys
import os

from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import ConnectivityFinder
from pymatgen.analysis.chemenv.connectivity.structure_connectivity import StructureConnectivity
from pymatgen.util.coord import get_angle
from pymatgen.core.structure import PeriodicSite, Structure
from typing import Generator, Sequence, Any, Union

## -----------------------------------------------------------------------------

def analyze_env(struc : Structure, mystrategy : str = "simple") -> tuple[list[int], StructureConnectivity]:
    '''
    Analyzes the coordination environments and returns the StructureConnectivity object for the crystal and the list of oxidation states.
    First, BVAnalyzer() calculates the oxidation states. Then, the LocalGeometryFinder() computes the structure_environment object, 
    from which the LightStructureEnvironment (LSE) is derived. Finally, The ConnectivityFinder() builds the StructureConnectivity (SE) based on LSE. 
    At the end only the SE is returned, as it includes the LSE object as an attribute.
    Parameters:
    ----------------
    struc : Structure 
        crystal Structure object from pymatgen
    mystrategy : string
	    The simple or combined strategy for calculating the coordination environments.
    '''
    if mystrategy == "simple":
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
    else:
        strategy = mystrategy
        
    # The BVAnalyzer class implements a maximum a posteriori (MAP) estimation method to determine oxidation states in a structure.
    bv = BVAnalyzer()
    oxid_states = bv.get_valences(struc)
    
    # Backup current stdout
    old_stdout = sys.stdout
    # Avoid printing to the console 
    sys.stdout = open(os.devnull, "w")
    # Print a long stroy every time it is initiated
    lgf = LocalGeometryFinder() 
    # Reset old stdout
    sys.stdout = old_stdout
    
    lgf.setup_structure(structure = struc)
    
    # Get the StructureEnvironments 
    se = lgf.compute_structure_environments(
        only_cations          = True,
        valences              = oxid_states,
        additional_conditions = [AdditionalConditions.ONLY_ANION_CATION_BONDS])
    
    # Get LightStructureEnvironments
    lse = LightStructureEnvironments.from_structure_environments(
        strategy               = strategy,
        structure_environments = se)

    # Get StructureConnectivity object
    cf = ConnectivityFinder()
    sc = cf.get_structure_connectivity(light_structure_environments = lse)

    return oxid_states, sc

## -----------------------------------------------------------------------------

def chunker(seq : Sequence, size : int) -> Generator[Any, None, None]:
    '''
    Chunks the indices of an iterable object to pieces of a given size.
    Parameters:
    ----------------
    seq : Iterable 
        The iterable object (list, tuple etc) to be chunked in smaller pieces.
    size : int
        the length of each chunk (not necessarily the last one).
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

## -----------------------------------------------------------------------------

def firstDegreeFeatures(SC_object : StructureConnectivity, oxidation_list : list[int], struct : Structure) -> dict:
    '''
    Calculates the desired primary features (related to the atom and nearest neighbors) based on SC object, 
    returns them as a dictionary. These features are stored for each atom, under their structure index.
    Features Include: Oxidation number, type of ion, element, coordination for all atoms.
    Cation specific features are the local(coordination) env and nearest neighbor elements & distances.
    Parameters:
    ----------------
    SC_object : StructureConnectivity
    oxidation_list : list[int]
        A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
    struct : Structure
        crystal Structure object from pymatgen
    '''

    # Take lightStructureEnvironment Obj from StructureConnecivity Obj
    LSE = SC_object.light_structure_environments
    # Take coordination/local environments from lightStructureEnvironment Obj
    local_Envs_list = LSE.coordination_environments
    structure_data : dict = {}
    for atomIndex, atom in enumerate(LSE.neighbors_sets):
        
        structure_data[atomIndex] = {}
        structure_data[atomIndex]['oxidation'] = oxidation_list[atomIndex]
        
        if atom == None:
            # Save coordniates here 
            structure_data[atomIndex]['ion'    ] = 'anion'
            structure_data[atomIndex]['element'] = struct[atomIndex].species_string
            structure_data[atomIndex]['coords' ] = struct[atomIndex].coords
            # Skip further featurization. We're not analyzing envs with anions
            continue

        structure_data[atomIndex]['ion'         ] = 'cation'
        structure_data[atomIndex]['element'     ] = struct[atomIndex].species_string
        structure_data[atomIndex]['coords'      ] = struct[atomIndex].coords
        structure_data[atomIndex]['localEnv'    ] = local_Envs_list[atomIndex]
        structure_data[atomIndex]['NN_distances'] = []

        neighbors = atom[0].neighb_sites_and_indices
        for nb in neighbors:
            # Pymatgen bug-fix (PeriodicNeighbor cannot be serialized, need to convert to PeriodicSite)
            site = PeriodicSite.from_dict(nb['site'].as_dict())
            nb_element  = site.species_string
            nb_distance = site.distance_from_point(struct[atomIndex].coords)
            structure_data[atomIndex]['NN_distances'].append([nb_distance, nb_element])
    
    return structure_data

## -----------------------------------------------------------------------------

def nnnFeatures(SC_object : StructureConnectivity, struct : Structure, structure_data : dict) -> dict:
    '''
    Calculates the desired NNN features based on SC object, and addes them to a dictionary (of primary features).
    These features are stored for each atom, under their structure index.
    NNN features Include: Polhedral neighbor elements, distances, connectivity angles & types. 
    Parameters:
    ----------------
    SC_object : StructureConnectivity
    oxidation_list : list[int]
        A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
    struct : Structure
        crystal Structure object from pymatgen
    structure_data : dict
        A dictionary containing primary features of the crystal. The NNN features will be added under the same atom index.
    '''

    nodeS = SC_object.environment_subgraph().nodes()

    for node in nodeS:
        distances   = []
        node_angleS = []

        for edge in SC_object.environment_subgraph().edges(node, data=True):

            # NNN distance calculation
            distance      = struct[edge[2]["start"]].distance(struct[edge[2]["end"]], edge[2]["delta"])
            start_element = struct[edge[2]["start"]].species_string
            end_element   = struct[edge[2]["end"  ]].species_string

            # Can't see an order on which side edge starts
            if node.atom_symbol != end_element:
                neighbor_element = end_element
            else:
                # This way if the 2 elements are different, the other name is saved.
                neighbor_element = start_element

            distance = [distance, neighbor_element]
            # Record as distance for this edge (NNN) and for this node (atom of interest)
            distances.append(distance)


            # NNN angles calculation
            ligandS = edge[2]["ligands"]

            connectivity = {}
            if len(ligandS) == 0:
                connectivity['kind'] = "noConnection"
            if len(ligandS) == 1:
                connectivity['kind'] = "corner"
            elif len(ligandS) == 2:
                connectivity['kind'] = "edge"
            elif len(ligandS) >= 3:
                connectivity['kind'] = "face"
            else:
                print('There was a problem with the connectivity.')

            edge_angleS : list[Union[list, dict]]= []
            edge_angleS.append(connectivity)
            for ligand in ligandS:
                pos0=struct[ligand[1]["start"]].frac_coords
                pos1=struct[ligand[1]["end"]].frac_coords+ligand[1]["delta"]
                cart_pos0 = struct.lattice.get_cartesian_coords(pos0)
                cart_pos1 = struct.lattice.get_cartesian_coords(pos1)

                pos2=struct[ligand[2]["start"]].frac_coords
                pos3=struct[ligand[2]["end"]].frac_coords+ligand[2]["delta"]
                cart_pos2 = struct.lattice.get_cartesian_coords(pos2)
                cart_pos3 = struct.lattice.get_cartesian_coords(pos3)
                
                angle = get_angle(cart_pos0-cart_pos1, cart_pos2-cart_pos3, units="degrees")

                if edge[2]['start'] != node.isite:
                    poly_nb = structure_data[edge[2]['start']]['element']
                else:
                    poly_nb = structure_data[edge[2]['end']]['element']  

                edge_angleS.append([angle, poly_nb])

            node_angleS.append(edge_angleS)

        structure_data[node.isite]['poly_distances']=distances 
        structure_data[node.isite]['connectivity_angles']=node_angleS
    
    return structure_data

## -----------------------------------------------------------------------------

def crysFeaturizer(SC_object : StructureConnectivity, oxidation_list : list[int]) -> dict:
    '''
    Calls firstDegreeFeatures() & nnnFeatures() functions to calculate the desired features 
    based on SC object, returns them as a dictionary. These features are stored for each atom,
    under their structure index.
    Features Include: Oxidation number, type of ion, element, coordination for all atoms.
    Cation specific features are the local(coordination) env, nearest neighbor elements & distances, 
    polhedral neighbor elements, distances, connectivity angles & types. 
    Parameters:
    ----------------
    SC_object : StructureConnectivity
    oxidation_list : list[int]
        A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
    '''
    struct = SC_object.light_structure_environments.structure #takes structure from StructureConnecivity Obj

    # Computing first degree features
    first_structure_data : dict = firstDegreeFeatures(SC_object=SC_object, oxidation_list = oxidation_list, struct=struct)
    # Add NNN features
    structure_data : dict = nnnFeatures(SC_object = SC_object, struct = struct, structure_data = first_structure_data)

    return structure_data

