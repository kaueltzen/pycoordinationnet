import pytorch_lightning as pl

from .model_transformer_lit import LitModelTransformer, LitCryturesData, LitProgressBar, LitMetricTracker

## ----------------------------------------------------------------------------

class GeoformerData(LitCryturesData):
    pass

## ----------------------------------------------------------------------------

class Geoformer(LitModelTransformer):
    def __init__(self,
            patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
            **kwargs):
        super().__init__(**kwargs)

        self.matric_tracker      = LitMetricTracker()
        self.early_stopping      = pl.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = 1, monitor = 'val_loss', mode = 'min')

        self.trainer     = pl.Trainer(
            max_epochs           = max_epochs,
            accelerator          = accelerator,
            devices              = devices,
            enable_checkpointing = True,
            enable_progress_bar  = True,
            logger               = False,
            strategy             = strategy,
            callbacks            = [LitProgressBar(), self.early_stopping, self.checkpoint_callback, self.matric_tracker])
    
    def train_model(self, data : GeoformerData):

        if type(data) != GeoformerData:
            raise ValueError('Data must be given as GeoformerData')

        # Train model on train data and use validation data for early stopping
        self.trainer.fit(self, data)

        # Get best model
        self.model = self.load_from_checkpoint(self.checkpoint_callback.best_model_path).model

        return self.checkpoint_callback.best_model_score

    def train_model_and_test(self, data : GeoformerData):
        # Train model
        best_val_score = self.train_model(data)
        # Test model
        self.trainer.test(self, data)

        return best_val_score, self.test_y, self.test_y_hat
