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