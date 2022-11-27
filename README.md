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

Note that features are only computed for cations:
```python
>>> features[63]['ion']
'cation'
>>> features[63].keys()
dict_keys(['oxidation', 'ion', 'element', 'coords', 'distances', 'ce', 'ce_distances', 'ce_angles'])
>>> features[64]['ion']
'anion'
>>> features[64].keys()
dict_keys(['oxidation', 'ion', 'element', 'coords'])
```

### Distances and angles

The distances to the nearest neighboring atoms are stored in the *distances* item:
```python
>>> features[0]['distances']
[[2.341961770123179, 'O'], [2.341961770123178, 'O'], [2.4321915059077597, 'O'], [2.341961770123179, 'O'], [2.43219150590776, 'O'], [2.3419617701231785, 'O'], [2.43219150590776, 'O'], [2.43219150590776, 'O']]
```
We see that since we have a *C:8* coordination environment that there are 8 nearest neighbors.

The *ce_distance* item contains the distances of the central atom to sites connected to the coordination environment.


## References

[1] Waroquiers, David, et al. "ChemEnv: a fast and robust coordination environment identification tool." Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials 76.4 (2020): 683-695.

[2] Pinsky, Mark, and David Avnir. "Continuous symmetry measures. 5. The classical polyhedra." Inorganic chemistry 37.21 (1998): 5575-5582.
