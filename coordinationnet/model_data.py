import torch

from typing import Any
from pymatgen.core.structure import Structure

from .features_datatypes import CoordinationFeatures

## ----------------------------------------------------------------------------

class CoordinationFeaturesData(torch.utils.data.Dataset):

    def __init__(self, X : list[Any], y = None) -> None:

        if not type(X) == list:
            raise ValueError(f'X must be of type list, but got type {type(X)}')

        if y is None:
            self.y = len(X)*[None]
        else:
            if isinstance(y, torch.Tensor):
                self.y = y
            else:
                self.y = torch.tensor(y)

        self.X = X

        for i, item in enumerate(X):
            if   isinstance(item, Structure):
                self.X[i] = CoordinationFeatures.from_structure(item, encode=True)
            elif isinstance(item, CoordinationFeatures):
                if not item.encoded:
                    self.X[i] = item.encode()
            else:
                raise ValueError(f'Items in X must be of type CoordinationFeatures or Structure, but item {i} is of type {type(item)}')

    def __len__(self) -> int:
        return len(self.X)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        return self.X[index], self.y[index]
