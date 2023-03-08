import dill
import torch
import pytorch_lightning as pl

from copy import deepcopy

from .model_data import CoordinationFeaturesData
from .model_lit  import LitCoordinationNet, LitCoordinationFeaturesData, LitProgressBar, LitMetricTracker

## ----------------------------------------------------------------------------

class CoordinationNet:

    def __init__(self,
            # Trainer options
            patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
            # Data options
            val_size = 0.1, batch_size = 128, num_workers = 2,
            # Model options
            **kwargs):

        self.lit_model           = LitCoordinationNet(**kwargs)
        self.lit_trainer         = None
        self.lit_trainer_options = {
            'patience'    : patience,
            'max_epochs'  : max_epochs,
            'accelerator' : accelerator,
            'devices'     : devices,
            'strategy'    : strategy,
        }
        self.lit_data_options    = {
            'val_size'    : val_size,
            'batch_size'  : batch_size,
            'num_workers' : num_workers,
        }

    def _setup_trainer_(self):
        self.lit_matric_tracker      = LitMetricTracker()
        self.lit_early_stopping      = pl.callbacks.EarlyStopping(monitor = 'val_loss', patience = self.lit_trainer_options['patience'])
        self.lit_checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = 1, monitor = 'val_loss', mode = 'min')

        self.lit_trainer = pl.Trainer(
            enable_checkpointing = True,
            enable_progress_bar  = True,
            logger               = False,
            max_epochs           = self.lit_trainer_options['max_epochs'],
            accelerator          = self.lit_trainer_options['accelerator'],
            devices              = self.lit_trainer_options['devices'],
            strategy             = self.lit_trainer_options['strategy'],
            callbacks            = [LitProgressBar(), self.lit_early_stopping, self.lit_checkpoint_callback, self.lit_matric_tracker])

    def cross_validation(self, data : CoordinationFeaturesData, n_splits, shuffle = True, random_state = 42):

        if not isinstance(data, CoordinationFeaturesData):
            raise ValueError('Data must be given as CoordinationFeaturesData')

        if n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        data  = LitCoordinationFeaturesData(data, self.lit_model.model.model_config, n_splits = n_splits, shuffle = shuffle, random_state = random_state, **self.lit_data_options)

        return self._cross_validataion(data)

    def _cross_validation(self, data : LitCoordinationFeaturesData):

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.lit_model.model

        for fold in range(data.n_splits):

            print(f'Training fold {fold+1}/{data.n_splits}...')
            data.setup_fold(fold)

            # Clone model
            self.lit_model.model = deepcopy(initial_model)

            # Train model
            best_val_score = self._train(data)['best_val_error']

            # Test model
            self.lit_trainer.test(self.lit_model, data)
            test_y, test_y_hat = self.lit_model.test_y, self.lit_model.test_y_hat

            # Print score
            print(f'Best validation score: {best_val_score}')

            # Save predictions for model evaluation
            y_hat = torch.cat((y_hat, test_y_hat))
            y     = torch.cat((y    , test_y    ))

        # Compute final test score
        test_loss = self.lit_model.loss(y_hat, y).item()

        return test_loss, y, y_hat

    def train(self, data : CoordinationFeaturesData):

        data = LitCoordinationFeaturesData(data, self.lit_model.model.model_config, **self.lit_data_options)

        return self._train(data)

    def _train(self, data : LitCoordinationFeaturesData):

        # We always need a new trainer for training the model
        self._setup_trainer_()

        # Train model on train data and use validation data for early stopping
        self.lit_trainer.fit(self.lit_model, data)

        # Get best model
        self.lit_model = self.lit_model.load_from_checkpoint(self.lit_checkpoint_callback.best_model_path)

        result = {
            'best_val_error': self.lit_checkpoint_callback.best_model_score.item(),
            'train_error'   : torch.stack(self.lit_matric_tracker.train_error).tolist(),
            'val_error'     : torch.stack(self.lit_matric_tracker.val_error  ).tolist() }

        return result

    def predict(self, data : CoordinationFeaturesData):

        data = LitCoordinationFeaturesData(data, self.lit_model.model.model_config, **self.lit_data_options)

        return self._predict(data)

    def _predict(self, data : LitCoordinationFeaturesData):

        if self.lit_trainer is None:
            self._setup_trainer_()

        y_hat_batched = self.lit_trainer.predict(self.lit_model, data)

        return torch.cat(y_hat_batched, dim=0)

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
