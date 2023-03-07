import torch
import pytorch_lightning as pl

from copy import deepcopy

from .model_transformer     import CryturesData
from .model_transformer_lit import LitModelTransformer, LitCryturesData, LitProgressBar, LitMetricTracker

## ----------------------------------------------------------------------------

class GeoformerData(CryturesData):
    pass

## ----------------------------------------------------------------------------

class Geoformer:
    def __init__(self,
            # Trainer options
            patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
            # Data options
            val_size = 0.1, batch_size = 128, num_workers = 2, shuffle = True, random_state = 42,
            # Model options
            **kwargs):

        self.lit_model           = LitModelTransformer(**kwargs)
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
            'shuffle'     : shuffle,
            'random_state': random_state,
        }

    def cross_validataion(self, data : GeoformerData, n_splits):

        if type(data) != GeoformerData:
            raise ValueError('Data must be given as GeoformerData')

        data = LitCryturesData(data, self.lit_model.model.model_config, n_splits = n_splits, **self.lit_data_options)

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.lit_model.model

        for fold in range(data.n_splits):

            print(f'Training fold {fold+1}/{data.n_splits}...')
            data.setup_fold(fold)

            # Clone model
            self.lit_model.model = deepcopy(initial_model)

            # Train model
            best_val_score     = self.train(data)

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

    def train(self, data):

        if type(data) is not LitCryturesData:
            data = LitCryturesData(data, self.lit_model.model.model_config, **self.lit_data_options)

        # We always need a new trainer for training the model
        self._setup_trainer_()

        # Train model on train data and use validation data for early stopping
        self.lit_trainer.fit(self.lit_model, data)

        # Get best model
        self.lit_model = self.lit_model.load_from_checkpoint(self.lit_checkpoint_callback.best_model_path)

        return self.lit_checkpoint_callback.best_model_score

    def predict(self, data):

        if type(data) is not LitCryturesData:
            data = LitCryturesData(data, self.lit_model.model.model_config, **self.lit_data_options)

        if self.lit_trainer is None:
            self._setup_trainer_()

        return self.lit_trainer.predict(self.lit_model, data)
