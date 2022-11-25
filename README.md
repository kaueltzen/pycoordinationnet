## CRYstal feaTURES (CRYTURES)

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

The *features* object is a dict object with as many items as there are atoms in the material. The items can be accessed with the atom/site index. Each item contains the oxidation state of the atom (*oxidation*), the local environments (*localEnv*), the nearest neighbor distances (*NN_distances*), the poly distances (*poly_distances*), and the angles between connecting atoms (*connectivity_angles*). Note that complex strategies can return multiple local environments for each site.

### Coordination environments

The coordination environment of a site is accessed as follows:
```python
>>> isite = 0
>>> features[isite]['localEnv']
[{'ce_symbol': 'C:8', 'ce_fraction': 1.0, 'csm': 2.217881949143581, 'permutation': [2, 4, 0, 1, 7, 5, 3, 6]}]
```
The *ce_symbol* specifies the symbol of the coordination environment (*ce*) as defined in the supplementary material of [1]. Each coordination environment is attributed a fraction given by the *ce_fraction* item. Since we have only a single coordination environment in this example, we have a *ce_fraction* of one.

## References

[1] Waroquiers, David, et al. "ChemEnv: a fast and robust coordination environment identification tool." Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials 76.4 (2020): 683-695.