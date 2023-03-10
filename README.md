## CoordinationNet

CoordinationNet is a transformer model that uses coordination information to predict materials properties. It is implemented in pytorch/lightning and provides a simple interface for training, predicting and cross-validation.

### Model initialization

The following code creates a new model instance with default settings:
```python
from coordinationnet import CoordinationNet

model = CoordinationNet()
```

### Creating a data set

As an example, we retrieve experimental and theoretical oxides from the materials project
```python
import numpy as np

from coordinationnet import mp_icsd_query, mp_icsd_clean

mats = mp_icsd_query("Q0tUKnAE52sy7hVO", experimental_data = False)
mats = mp_icsd_clean(mats)
# Remove 'mp-554015' due to bug #2756
mats = np.delete(mats, 3394)
# Need to convert numpy array to list for serialization
mats = mats.tolist()
```

We extract structures and target values (formation energy) from our materials list:
```python
structures = [  mat['structure']                  for mat in mats ]
targets    = [ [mat['formation_energy_per_atom']] for mat in mats ]
```

For training the network or making predictions, we must convert structures to cooridnation features. This is achieved using:
```python
from coordinationnet import CoordinationFeaturesData

data = CoordinationFeaturesData(structures, y = targets, verbose = True)
```
Note that *y* (i.e. the target values) is optional and can be left empty if the network is not trained and only used for making predictions on this data.

### Run cross-validation
CoordinationNet implements a cross-validation method that can be easily used:
```python
from monty.serialization import dumpfn

mae, y, y_hat = model.cross_validation(data, 10)

print('Final MAE:', mae)

# Save result
dumpfn({'y_hat': y_hat.tolist(),
        'y'    : y    .tolist(),
        'mae'  : mae },
        'eval-test.txt')
```

### Train and predict
Model training and computing predictions:
```python
import matplotlib.pyplot as plt

result = model.train(data)

plt.plot(result['train_error'])
plt.plot(result['val_error'])
plt.show()

model.predict(data)
```

### Save and load model
A trained model can be easily saved and loaded using:
```python
model.save('model.dill')
model = CoordinationNet.load('model.dill')
```

### Advanced model initialization
CoordinationNet has a modular structure so that individual components can be included or excluded from the model. Which model components are included is controlled using the *CoordinationNetConfig*. The following shows the default values for CoordinationNet:
```python
from coordinationnet import CoordinationNetConfig

model_config = CoordinationNetConfig(
    composition           = False,
    sites                 = False,
    sites_oxid            = False,
    sites_ces             = False,
    site_features         = True,
    site_features_ces     = True,
    site_features_oxid    = True,
    site_features_csms    = True,
    site_features_ligands = False,
    ligands               = False,
    ce_neighbors          = False,
)
```
If a value is not specified, it will be set to *False* by default. Hence, the above configuration is equivalen to
```python
model_config = CoordinationNetConfig(
    site_features         = True,
    site_features_ces     = True,
    site_features_oxid    = True,
    site_features_csms    = True,
)
```

The following code creates a new model instance with additional keyword arguments:
```python
from coordinationnet import CoordinationNet

model = CoordinationNet(
    # Model components
    model_config = model_config,
    # Dense layer options
    layers = [200, 4096, 1024, 512, 128, 1], dropout = 0.0, skip_connections = False, batchnorm = False,
    # Transformer options
    edim = 200, nencoders = 4, nheads = 4, dropout_transformer = 0.0, dim_feedforward = 200,
    # Data options
    batch_size = 128, num_workers = 10,
    # Optimizer options
    scheduler = 'plateau', devices=[2], patience = 2, lr = 1e-4, max_epochs = 1000)

```
All keyword arguments correspond to default values and don't have to be specified unless changed.

## Coordination Features

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
from coordinationnet import CoordinationFeatures

features = CoordinationFeatures.from_structure(structure)
```

The *features* object is a dict object with as many items as there are atoms in the material. The items can be accessed with the atom/site index. Each item contains the oxidation state of the atom (*oxidation*), the local environments (*ce*), the nearest neighbor distances (*distances*), the distances to neighboring coordination environments (*ce_distances*), and the angles between coordination environments (*ce_angles*). Note that complex strategies can return multiple local environments for each site.

### Coordination environments

The coordination environment of the first site (isite = 0) is accessed as follows:
```python
>>> features.get_site_features(0)['ce']
[{'ce_symbol': 'C:8', 'ce_fraction': 1.0, 'csm': 2.217881949143581, 'permutation': [2, 4, 0, 1, 7, 5, 3, 6]}]
```
The *ce_symbol* specifies the symbol of the coordination environment (*ce*) as defined in the supplementary material of *Waroquiers et al. 2020* [1]. In this example we have *C:8* which refers to a cube where the central site has 8 neighbors. Each coordination environment is attributed a fraction given by the *ce_fraction* item. Since we have only a single coordination environment in this example, we have a *ce_fraction* of one. The Continuous Symmetry Measure (*CSM*) specifies the distance of the coordination environment to the perfect model environment (given by the *csm* item) [2]. The CSM value ranges between zero and 100, where a value of zero represents a perfect match. To compute the similarity of the coordination environment, all possible permutations of neighboring sites must be tested. The permutation with the minimal CSM is given by the *permutation* item.

Note that features are only computed for cations, because they form the centers of coordination environments:
```python
>>> features.get_site_features(63)['ion']
'cation'
>>> features.get_site_features(63).keys()
dict_keys(['oxidation', 'ion', 'element', 'coordinates', 'distances', 'ce', 'ce_distances', 'ce_angles'])
>>> features.get_site_features(64)['ion']
'anion'
>>> features.get_site_features(64).keys()
dict_keys(['oxidation', 'ion', 'element', 'coordinates'])
```

### Distances and angles

The distances between the center atom to the nearest neighboring atoms (ligands) are stored in the *distances* item:
```python
>>> features.get_site_features(0)['distances']
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

### Encoded features

Features contain several categorical variables, for instance element names type of angles between coordination environments (i.e. corner, face, etc.). These variables must be encoded such that they can be used as inputs to machine learning models. The simples approach is to replace categorical varibales by integer indices, which enables us to use embeddings for categorical variables. Also the value of oxidation states is not very well suited for machine learning models, which are positive and negative integer values. We also recode oxidation states as positive integers, such that embeddings can be used.

Features can be encoded and decoded as follows:
```python
features = features.encode()
features = features.decode()
```

## References

[1] Waroquiers, David, et al. "ChemEnv: a fast and robust coordination environment identification tool." Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials 76.4 (2020): 683-695.

[2] Pinsky, Mark, and David Avnir. "Continuous symmetry measures. 5. The classical polyhedra." Inorganic chemistry 37.21 (1998): 5575-5582.
