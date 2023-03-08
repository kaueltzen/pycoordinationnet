import numpy as np
import torch

from monty.serialization import loadfn

## ----------------------------------------------------------------------------

class CryturesData(torch.utils.data.Dataset):

    def __init__(self, filename) -> None:

        X, y = self.__load_data__(filename)

        if isinstance(y, torch.Tensor):
            self.y = y
        else:
            self.y = torch.tensor(y)

        self.X = X

        for i, crytures in enumerate(X):
            if not crytures.encoded:
                self.X[i] = crytures.encode()

    @classmethod
    def __load_data__(self, filename):
        data = loadfn(filename)

        # Get features and target values
        X = [  x['features']                   for x in data ]
        y = [ [x['formation_energy_per_atom']] for x in data ]

        # Filter outliers
        X = np.array(X)[ (np.array(y) > -6)[:,0] ].tolist()
        y = np.array(y)[ (np.array(y) > -6)[:,0] ].tolist()

        return X, y

    def __len__(self):
        return len(self.X)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        return self.X[index], self.y[index]
