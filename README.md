## CRYstal feaTURES (CRYTURES)

This packages uses *pymatgen* to compute coordination environments, their distances and angles. The output is such that it can be used as features for machine learning models.

A coordination environment consists of a central atom or ion, which is usually metallic and is called the coordination center. It is surrounded by several bounding ions that are called ligands. The position of ligands defines the shape of the coordination environment, which is described in terms of the coordination number (number of ligands) and a symbol for the shape of the environment (e.g. C for cubic)

### Getting started

First, we download a material from the materials project:
```python
from pymatgen.ext.matproj import MPRester

mid = 'mp-12236'

with MPRester("Q0tUKnAE52sy7hVO") as m:
    structure = m.get_structure_by_material_id(mid, conventional_unit_cell=True)
```

The crystal features can be computed directly from the structure object:
```python
from crytures import featurize

features = featurize(structure)
```

The *features* object is a dict object with as many items as there are atoms in the material. The items can be accessed with the atom/site index. Each item contains the oxidation state of the atom (*oxidation*), the local environments (*ce*), the nearest neighbor distances (*distances*), the distances to neighboring coordination environments (*ce_distances*), and the angles between coordination environments (*ce_angles*). Note that complex strategies can return multiple local environments for each site.

### Coordination environments

The coordination environment of a site is accessed as follows:
```python
>>> isite = 0
>>> features[isite]['ce']
[{'ce_symbol': 'C:8', 'ce_fraction': 1.0, 'csm': 2.217881949143581, 'permutation': [2, 4, 0, 1, 7, 5, 3, 6]}]
```
The *ce_symbol* specifies the symbol of the coordination environment (*ce*) as defined in the supplementary material of *Waroquiers et al. 2020* [1]. In this example we have *C:8* which refers to a cube where the central site has 8 neighbors. Each coordination environment is attributed a fraction given by the *ce_fraction* item. Since we have only a single coordination environment in this example, we have a *ce_fraction* of one. The Continuous Symmetry Measure (*CSM*) specifies the distance of the coordination environment to the perfect model environment (given by the *csm* item) [2]. The CSM value ranges between zero and 100, where a value of zero represents a perfect match. To compute the similarity of the coordination environment, all possible permutations of neighboring sites must be tested. The permutation with the minimal CSM is given by the *permutation* item.

Note that features are only computed for cations, because they form the centers of coordination environments:
```python
>>> features[63]['ion']
'cation'
>>> features[63].keys()
dict_keys(['oxidation', 'ion', 'element', 'coordinates', 'distances', 'ce', 'ce_distances', 'ce_angles'])
>>> features[64]['ion']
'anion'
>>> features[64].keys()
dict_keys(['oxidation', 'ion', 'element', 'coordinates'])
```

### Distances and angles

The distances between the center atom to the nearest neighboring atoms (ligands) are stored in the *distances* item:
```python
>>> features[0]['distances']
[('Er', 'O', 2.341961770123179), ('Er', 'O', 2.341961770123178), ('Er', 'O', 2.4321915059077597), ('Er', 'O', 2.341961770123179), ('Er', 'O', 2.43219150590776), ('Er', 'O', 2.3419617701231785), ('Er', 'O', 2.43219150590776), ('Er', 'O', 2.43219150590776)]
```
We see that since we have a *C:8* coordination environment that there are 8 nearest neighbors. The first two elements of each tuple denote the atom types from which the distance is measured. The distance is the third entry of each tuple.

The *ce_distance* item contains the distances to neighboring coordination environments, always measured from the center atoms.
```python
[('Er', 'Er', 3.781941127654686), ('Er', 'Er', 3.781941127654686), ('Er', 'Er', 3.781941127654686), ('Er', 'Er', 3.781941127654686), ('Er', 'Ga', 3.781941127654686), ('Er', 'Ga', 3.781941127654686), ('Er', 'Ga', 3.781941127654686), ('Er', 'Ga', 3.087942), ('Er', 'Ga', 3.452424111288328), ('Er', 'Ga', 3.452424111288328), ('Er', 'Ga', 3.087942), ('Er', 'Ga', 3.781941127654686), ('Er', 'Ga', 3.452424111288328), ('Er', 'Ga', 3.452424111288328)]
```
The first two entries of each tuple contain the elements of the center atoms. The third entry is the distance between center atoms.

Similarly, the *ce_angles* item contains the angle between neighboring coordination environments.
```python
[['edge', ('Er', 'O', 'Er', 104.76176533545028), ('Er', 'O', 'Er', 104.76176533545029)], ['edge', ('Er', 'O', 'Er', 104.76176533545029), ('Er', 'O', 'Er', 104.76176533545028)], ['edge', ('Er', 'O', 'Er', 104.76176533545026), ('Er', 'O', 'Er', 104.76176533545035)], ['edge', ('Er', 'O', 'Er', 104.76176533545026), ('Er', 'O', 'Er', 104.76176533545028)], ['corner', ('Er', 'O', 'Ga', 122.8127776876257)], ['corner', ('Er', 'O', 'Ga', 122.81277768762573)], ['corner', ('Er', 'O', 'Ga', 122.81277768762573)], ['edge', ('Er', 'O', 'Ga', 93.78531247833918), ('Er', 'O', 'Ga', 93.78531247833915)], ['edge', ('Er', 'O', 'Ga', 104.54130312850349), ('Er', 'O', 'Ga', 101.4103174150423)], ['edge', ('Er', 'O', 'Ga', 104.54130312850347), ('Er', 'O', 'Ga', 101.41031741504229)], ['edge', ('Er', 'O', 'Ga', 93.78531247833916), ('Er', 'O', 'Ga', 93.78531247833918)], ['corner', ('Er', 'O', 'Ga', 122.81277768762573)], ['edge', ('Er', 'O', 'Ga', 104.54130312850349), ('Er', 'O', 'Ga', 101.41031741504227)], ['edge', ('Er', 'O', 'Ga', 104.54130312850346), ('Er', 'O', 'Ga', 101.41031741504229)]]
```
Each tuple contains the three atom types along which the angle is measured. The second entry is the type of the ligand, while the first and the third are the center atoms of the coordination environments. The fourth entry is the angle at the ligand.

Note that distances and angles are computed for all neighbors by considering symmetries stemming from the periodic boundary conditions.

## References

[1] Waroquiers, David, et al. "ChemEnv: a fast and robust coordination environment identification tool." Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials 76.4 (2020): 683-695.

[2] Pinsky, Mark, and David Avnir. "Continuous symmetry measures. 5. The classical polyhedra." Inorganic chemistry 37.21 (1998): 5575-5582.
