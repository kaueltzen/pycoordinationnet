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

from typing                  import Union
from copy                    import deepcopy
from sklearn.model_selection import KFold

from .model_config           import CoordinationNetConfig, DefaultCoordinationNetConfig
from .model_data             import CoordinationFeaturesData
from .model_transformer      import ModelCoordinationNet
from .model_transformer_data import BatchedCoordinationFeaturesData, CoordinationFeaturesLoader
from .model_gnn              import ModelGraphCoordinationNet
from .model_gnn_data         import GraphCoordinationFeaturesLoader
from .model_lit              import LitModel, LitDataset

## ----------------------------------------------------------------------------

#torch.multiprocessing.set_sharing_strategy('file_system')

## ----------------------------------------------------------------------------

class LitCoordinationFeaturesData(LitDataset):
    def __init__(self, data : CoordinationFeaturesData, val_size = 0.2, batch_size = 32, num_workers = 2):
        super().__init__(data, val_size = val_size, batch_size = batch_size, num_workers = num_workers)

    # Custom method to create a data loader
    def get_dataloader(self, data):
        return CoordinationFeaturesLoader(data, batch_size = self.batch_size, num_workers = self.num_workers)

## ----------------------------------------------------------------------------

class CoordinationNet:

    def __init__(self, cache_file = None, **kwargs):

        self.lit_model  = LitModel(ModelCoordinationNet, **kwargs)
        self.cache_file = cache_file

        if self.lit_model.global_rank == 0:

            print(f'{model_config}')

            print(f'Creating a GNN model with {self.n_parameters:,} parameters')

    def fit_scaler(self, data : LitCoordinationFeaturesData):
        y = torch.cat([ y_batch for _, y_batch in data.get_dataloader(data.data) ])
        self.lit_model.model.scaler_outputs.fit(y)

    def train(self, data : Union[CoordinationFeaturesData, BatchedCoordinationFeaturesData]):

        data = self.prepare_data(data)
        data = LitCoordinationFeaturesData(data, **self.lit_model.data_options)

        # Fit scaler to target values. The scaling of model outputs is done
        # by the model itself
        self.fit_scaler(data)

        self.lit_model, stats = self.lit_model._train(data)

        return stats

    def test(self, data : Union[CoordinationFeaturesData, BatchedCoordinationFeaturesData]):

        data = self.prepare_data(data)
        data = LitCoordinationFeaturesData(data, **self.lit_model.data_options)

        return self.lit_model._test(data)

    def predict(self, data):

        data = self.prepare_data(data)
        data = LitCoordinationFeaturesData(data, **self.lit_model.data_options)

        return self.lit_model._predict(data)

    def cross_validation(self, data, n_splits, shuffle = True, random_state = 42):

        if n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        data  = self.prepare_data(data)

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.lit_model

        for fold, (index_train, index_test) in enumerate(KFold(n_splits, shuffle = shuffle, random_state = random_state).split(data)):

            if self.lit_model.global_rank == 0:
                print(f'Training fold {fold+1}/{n_splits}...')

            data_train = torch.utils.data.Subset(data, index_train)
            data_test  = torch.utils.data.Subset(data, index_test )

            # Clone model
            self.lit_model = deepcopy(initial_model)

            # Train model
            best_val_score = self.train(data_train)['best_val_error']

            # Test model
            test_y, test_y_hat, _ = self.test(data_test)

            # Print score
            if self.lit_model.global_rank == 0:
                print(f'Best validation score: {best_val_score}')

            # Save predictions for model evaluation
            y_hat = torch.cat((y_hat, test_y_hat))
            y     = torch.cat((y    , test_y    ))

        # Reset model
        self.lit_model = initial_model

        # Compute final test score
        test_loss = self.lit_model.loss(y_hat, y).item()

        return test_loss, y, y_hat

    def prepare_data(self, data):

        if isinstance(data, torch.utils.data.Subset):
            if not isinstance(data.dataset, BatchedCoordinationFeaturesData):
                raise ValueError(f'Data Subset must contain dataset of type BatchedCoordinationFeaturesData, but got type {type(data.dataset)}')

        else:
            if not (isinstance(data, CoordinationFeaturesData) or isinstance(data, BatchedCoordinationFeaturesData)):
                raise ValueError(f'Data must be given as CoordinationFeaturesData or BatchedCoordinationFeaturesData, but got type {type(data)}')

            if isinstance(data, CoordinationFeaturesData):
                data = BatchedCoordinationFeaturesData(data, self.lit_model.model.model_config, self.lit_model.data_options['batch_size'], cache_file = self.cache_file)

        return data

    @classmethod
    def load(cls, filename : str) -> 'CoordinationNet':

        with open(filename, 'rb') as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(f'file {filename} contains incorrect model class {type(model)}')

        return model

    def save(self, filename : str) -> None:

        with open(filename, 'wb') as f:
            dill.dump(self, f)

## ----------------------------------------------------------------------------

class LitGraphCoordinationFeaturesData(LitDataset):
    def __init__(self, data : CoordinationFeaturesData, val_size = 0.2, batch_size = 32, num_workers = 2):
        super().__init__(data, val_size = val_size, batch_size = batch_size, num_workers = num_workers)

    # Custom method to create a data loader
    def get_dataloader(self, data):
        return GraphCoordinationFeaturesLoader(data, batch_size = self.batch_size, num_workers = self.num_workers)

## ----------------------------------------------------------------------------

class GraphCoordinationNet:

    def __init__(self, **kwargs):

        self.lit_model = LitModel(ModelGraphCoordinationNet, **kwargs)

        if self.lit_model.global_rank == 0:

            print(f'{self.lit_model.model.model_config}')

            print(f'Creating a GNN model with {self.lit_model.model.n_parameters:,} parameters')

    def fit_scaler(self, data : LitGraphCoordinationFeaturesData):
        y = torch.cat([ y_batch for _, y_batch in data.get_dataloader(data.data) ])
        self.lit_model.model.scaler_outputs.fit(y)

    def train(self, data : CoordinationFeaturesData):

        data = LitGraphCoordinationFeaturesData(data, **self.lit_model.data_options)

        # Fit scaler to target values. The scaling of model outputs is done
        # by the model itself
        #self.fit_scaler(data)

        self.lit_model, stats = self.lit_model._train(data)

        return stats

    def test(self, data : CoordinationFeaturesData):

        data = LitGraphCoordinationFeaturesData(data, **self.lit_model.data_options)

        return self.lit_model._test(data)

    def predict(self, data : CoordinationFeaturesData):

        data = LitGraphCoordinationFeaturesData(data, **self.lit_model.data_options)

        return self.lit_model._predict(data)

    def cross_validation(self, data, n_splits, shuffle = True, random_state = 42):

        if n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.lit_model

        for fold, (index_train, index_test) in enumerate(KFold(n_splits, shuffle = shuffle, random_state = random_state).split(data)):

            if self.lit_model.global_rank == 0:
                print(f'Training fold {fold+1}/{n_splits}...')

            data_train = torch.utils.data.Subset(data, index_train)
            data_test  = torch.utils.data.Subset(data, index_test )

            # Clone model
            self.lit_model = deepcopy(initial_model)

            # Train model
            best_val_score = self.train(data_train)['best_val_error']

            # Test model
            test_y, test_y_hat, _ = self.test(data_test)

            # Print score
            if self.lit_model.global_rank == 0:
                print(f'Best validation score: {best_val_score}')

            # Save predictions for model evaluation
            y_hat = torch.cat((y_hat, test_y_hat))
            y     = torch.cat((y    , test_y    ))

        # Reset model
        self.lit_model = initial_model

        # Compute final test score
        test_loss = self.lit_model.loss(y_hat, y).item()

        return test_loss, y, y_hat

    @classmethod
    def load(cls, filename : str) -> 'CoordinationNet':

        with open(filename, 'rb') as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(f'file {filename} contains incorrect model class {type(model)}')

        return model

    def save(self, filename : str) -> None:

        with open(filename, 'wb') as f:
            dill.dump(self, f)
