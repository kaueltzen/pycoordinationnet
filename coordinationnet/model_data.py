## Copyright (C) 2023 Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## ----------------------------------------------------------------------------

import dill
import torch
import numpy as np

from typing import Any
from pymatgen.core.structure import Structure

from .features_datatypes import CoordinationFeatures

## ----------------------------------------------------------------------------

class CoordinationFeaturesData(torch.utils.data.Dataset):

    def __init__(self, X : list[Any], y = None, verbose = False) -> None:

        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            raise ValueError(f'X must be of type list or numpy array, but got type {type(X)}')

        if y is None:
            self.y = len(X)*[None]
        else:
            if isinstance(y, torch.Tensor):
                self.y = y
            else:
                self.y = torch.tensor(y)

        self.X = np.array(X)

        for i, item in enumerate(self.X):
            if   isinstance(item, Structure):
                if verbose:
                    print(f'Featurizing structure {i+1}/{len(X)}')
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

    @classmethod
    def load(cls, filename : str) -> 'CoordinationFeaturesData':

        with open(filename, 'rb') as f:
            data = dill.load(f)

        if not isinstance(data, cls):
            raise ValueError(f'file {filename} contains incorrect data class {type(data)}')

        # For backward compatibility
        if isinstance(data.X, list):
            data.X = np.array(data.X)

        return data

    def save(self, filename : str) -> None:

        with open(filename, 'wb') as f:
            dill.dump(self, f)

## ----------------------------------------------------------------------------

class Batch():

    # This function is used by the estimator to push
    # data to GPU
    def to(self, device=None):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, 'to'):
                 result.__setattr__(attr, value.to(device=device))
        return result

    # This function will be called by the pytorch DataLoader
    # after collate_fn has assembled the batch
    def pin_memory(self):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, 'pin_memory'):
                 result.__setattr__(attr, value.pin_memory())
        return result

    def share_memory_(self):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, 'share_memory_'):
                 result.__setattr__(attr, value.share_memory_())
        return result
