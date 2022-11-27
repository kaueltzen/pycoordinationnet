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
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.util.coord     import get_angle
from typing import Union

## -----------------------------------------------------------------------------

def analyze_environment(structure : Structure, mystrategy : str = 'simple') -> tuple[StructureConnectivity, list[int]]:
    '''
    Analyzes the coordination environments and returns the StructureConnectivity object for the crystal and the list of oxidation states.
    First, BVAnalyzer() calculates the oxidation states. Then, the LocalGeometryFinder() computes the structure_environment object, 
    from which the LightStructureEnvironment (LSE) is derived. Finally, The ConnectivityFinder() builds the StructureConnectivity (SE) based on LSE. 
    At the end only the SE is returned, as it includes the LSE object as an attribute.

    Args:
        struc (Structure):
            crystal Structure object from pymatgen
        mystrategy (string):
	        The simple or combined strategy for calculating the coordination environments
    '''
    if mystrategy == 'simple':
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
    else:
        strategy = mystrategy
        
    # The BVAnalyzer class implements a maximum a posteriori (MAP) estimation method to determine oxidation states in a structure.
    bv = BVAnalyzer()
    oxid_states = bv.get_valences(structure)
    
    # Backup current stdout
    old_stdout = sys.stdout
    # Avoid printing to the console 
    sys.stdout = open(os.devnull, 'w')
    # Print a long stroy every time it is initiated
    lgf = LocalGeometryFinder() 
    # Reset old stdout
    sys.stdout = old_stdout
    
    lgf.setup_structure(structure = structure)
    
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

    return sc, oxid_states

## -----------------------------------------------------------------------------

def firstDegreeFeatures(structure_connectivity : StructureConnectivity, oxidation_list : list[int]) -> dict:
    '''
    Calculates the desired primary features (related to the atom and nearest neighbors) based on SC object, 
    returns them as a dictionary. These features are stored for each atom, under their structure index.
    Features Include: Oxidation number, type of ion, element, coordination for all atoms.
    Cation specific features are the local(coordination) env and nearest neighbor elements & distances.

    Args:
        structure_connectivity (StructureConnectivity):
            The connectivity structure of the material
        oxidation_list (list[int]):
            A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.

    Returns:
        A dictionary with first degree features
    '''

    structure = structure_connectivity.light_structure_environments.structure
    # Take lightStructureEnvironment Obj from StructureConnecivity Obj
    lse = structure_connectivity.light_structure_environments
    # Take coordination/local environments from lightStructureEnvironment Obj
    local_Envs_list = lse.coordination_environments
    structure_data : dict = {}
    for atomIndex, atom in enumerate(lse.neighbors_sets):
        
        structure_data[atomIndex] = {}
        structure_data[atomIndex]['oxidation'] = oxidation_list[atomIndex]
        
        if atom == None:
            # Save coordniates here 
            structure_data[atomIndex]['ion'    ] = 'anion'
            structure_data[atomIndex]['element'] = structure[atomIndex].species_string
            structure_data[atomIndex]['coords' ] = structure[atomIndex].coords
            # Skip further featurization. We're not analyzing envs with anions
            continue

        structure_data[atomIndex]['ion'         ] = 'cation'
        structure_data[atomIndex]['element'     ] = structure[atomIndex].species_string
        structure_data[atomIndex]['coords'      ] = structure[atomIndex].coords
        structure_data[atomIndex]['localEnv'    ] = local_Envs_list[atomIndex]
        structure_data[atomIndex]['NN_distances'] = []

        neighbors = atom[0].neighb_sites_and_indices
        for nb in neighbors:
            # Pymatgen bug-fix (PeriodicNeighbor cannot be serialized, need to convert to PeriodicSite)
            # (fixed with 0eb1e3d72fd894b7ba39a5129fbd8b18aedf4b46)
            site = PeriodicSite.from_dict(nb['site'].as_dict())
            nb_element  = site.species_string
            nb_distance = site.distance_from_point(structure[atomIndex].coords)
            structure_data[atomIndex]['NN_distances'].append([nb_distance, nb_element])
    
    return structure_data

## -----------------------------------------------------------------------------

def nnnFeatures(structure_connectivity : StructureConnectivity, structure_data : dict) -> dict:
    '''
    Calculates the desired NNN (next nearest neighbors) features based on SC object,
    and adds them to a dictionary (of primary features). These features are stored
    for each atom, under their structure index. NNN features Include: Polhedral neighbor
    elements, distances, connectivity angles & types. 

    Args:
        structure_connectivity (StructureConnectivity):
            The connectivity structure of the material
        structure_data (dict):
            A dictionary containing primary features of the crystal. The NNN features
            will be added under the same atom index.
    
    Returns:
        A dictionary with next nearest neighbor features added to the structure_data
        object
    '''

    structure = structure_connectivity.light_structure_environments.structure
    nodes     = structure_connectivity.environment_subgraph().nodes()

    # Loop over all sites in the structure
    for node in nodes:
        distances   = []
        node_angles = []

        for edge in structure_connectivity.environment_subgraph().edges(node, data=True):

            # NNN distance calculation
            distance      = structure[edge[2]['start']].distance(structure[edge[2]['end']], edge[2]['delta'])
            start_element = structure[edge[2]['start']].species_string
            end_element   = structure[edge[2]['end'  ]].species_string

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
            ligands = edge[2]['ligands']

            connectivity = ''
            if   len(ligands) == 0:
                connectivity = 'isolated'
            if   len(ligands) == 1:
                connectivity = 'corner'
            elif len(ligands) == 2:
                connectivity = 'edge'
            else:
                connectivity = 'face'

            edge_angles : list[Union[list, dict]] = []
            edge_angles.append(connectivity)
            for ligand in ligands:
                # Ligands/anions always have a higher atom index than cations. For the ligands list,
                # start will always have a lower atom index than end, which means that we always start
                # at cations and then go to the ligand.

                # We consider two connecting atoms of the ligand. Get the coordinates of all three
                # sites
                pos0 = structure[ligand[1]['start']].frac_coords
                pos1 = structure[ligand[1]['end'  ]].frac_coords + ligand[1]['delta']
                pos2 = structure[ligand[2]['start']].frac_coords
                pos3 = structure[ligand[2]['end'  ]].frac_coords + ligand[2]['delta']

                cart_pos0 = structure.lattice.get_cartesian_coords(pos0)
                cart_pos1 = structure.lattice.get_cartesian_coords(pos1)
                cart_pos2 = structure.lattice.get_cartesian_coords(pos2)
                cart_pos3 = structure.lattice.get_cartesian_coords(pos3)
                
                # Measure the angle at the ligand
                angle = get_angle(cart_pos0-cart_pos1, cart_pos2-cart_pos3, units='degrees')

                # Get the name of the element of the other connecting cation
                if edge[2]['start'] != node.isite:
                    poly_nb = structure_data[edge[2]['start']]['element']
                else:
                    poly_nb = structure_data[edge[2]['end'  ]]['element']  

                edge_angles.append([angle, poly_nb])

            node_angles.append(edge_angles)

        structure_data[node.isite]['poly_distances'     ] = distances 
        structure_data[node.isite]['connectivity_angles'] = node_angles
    
    return structure_data

## -----------------------------------------------------------------------------

def featurize(structure : Structure, env_strategy = 'simple') -> dict:
    '''
    Calls firstDegreeFeatures() & nnnFeatures() functions to calculate the desired features 
    based on SC object, returns them as a dictionary. These features are stored for each atom,
    under their structure index.
    Features Include: Oxidation number, type of ion, element, coordination for all atoms.
    Cation specific features are the local(coordination) env, nearest neighbor elements & distances, 
    polhedral neighbor elements, distances, connectivity angles & types. 

    Args:
        structure (Structure):
            A pymatgen structure object
        env_strategy (string):
            The strategy used for computing environments
    
    Returns:
        A dictionary of features for each atom in the structure
    '''
    structure_connectivity, oxid_states = analyze_environment(structure, mystrategy = env_strategy)

    # Computing first degree features
    first_structure_data : dict = firstDegreeFeatures(
        structure_connectivity = structure_connectivity,
        oxidation_list         = oxid_states)
    # Add NNN features
    structure_data : dict = nnnFeatures(
        structure_connectivity = structure_connectivity,
        structure_data         = first_structure_data)

    return structure_data
