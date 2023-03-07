
import numpy as np

from collections import OrderedDict

## ----------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience       = patience
        self.verbose        = verbose
        self.counter        = 0
        self.early_stop     = False
        self.val_loss_min   = None
        self.val_loss_round = None
        self.delta          = delta
        self.trace_func     = trace_func
        self.model_state    = None

    def __call__(self, val_loss, model):

        if np.isnan(val_loss):
            self.early_stop = True
            return self.early_stop

        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.val_loss_min - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
        
        return self.early_stop

    def save_model(self, model):
        self.model_state = OrderedDict()
        for key, value in model.state_dict().items():
            self.model_state[key] = value.detach().cpu().clone()

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'EarlyStopping: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.save_model(model)
        self.val_loss_min   = val_loss
        self.val_loss_round = val_loss
        self.counter        = 0

    def reset(self):
        self.counter        = 0
        self.early_stop     = False
        self.val_loss_round = None

    def reset_full(self):
        self.reset()
        self.val_loss_min   = None
        self.val_loss_round = None
        self.model_state    = None
