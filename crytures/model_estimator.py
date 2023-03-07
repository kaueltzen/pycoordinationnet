
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import train_test_split

from .model_stopper import EarlyStopper

## Model Estimator
## ----------------------------------------------------------------------------

class ModelEstimator:
    def __init__(self, model, dataloader, epochs=10000, patience=7, val_size=0.0, optimizer=torch.optim.Adam, optimargs={}, scheduler=None, schedargs = {}, loss_function=torch.nn.L1Loss(), random_state=42, batch_size=None, num_workers = 4, device=None, pin_memory=False, verbose=False):

        self.model         = model
        self.dataloader    = dataloader
        self.epochs        = epochs
        self.patience      = patience
        self.val_size      = val_size
        self.batch_size    = batch_size
        self.loss_function = loss_function
        self.optimizer     = optimizer
        self.optimargs     = optimargs
        self.scheduler     = scheduler
        self.schedargs     = schedargs
        self.num_workers   = num_workers
        self.device        = device
        self.pin_memory    = pin_memory
        self.random_state  = random_state
        self.verbose       = verbose

    def fit(self, dataset, **kwargs):
        return self(dataset, **kwargs)

    def __call__(self, dataset, loss_function=None):

        if loss_function is None:
            loss_function = self.loss_function

        optimizer, scheduler = self._get_optimizer()

        hist_train = []
        hist_val   = []

        if self.val_size > 0.0:
            dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1.0 - self.val_size, self.val_size])

        else:
            dataset_val = dataset_train

        if self.batch_size is None:
            self.batch_size = len(dataset)
            self.shuffle    = False

        if self.device is not None:
            self.model = self.model.to(device=self.device)

        loader_train = self.dataloader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        loader_val   = self.dataloader(dataset_val  , batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

        es = EarlyStopper(patience=self.patience, verbose=False)

        if self.epochs is None:
            self.epochs = sys.maxsize

        for _epoch in range(0, self.epochs):
            self.model.train()
            loss_train   = 0.0
            loss_train_n = 0.0
            for _, data in enumerate(loader_train, 0):
                # Get X (inputs) and y (targets)
                X_batch, y_batch = data
                # Push to cuda if required
                if self.device is not None:
                    X_batch = X_batch.to(device=self.device)
                    y_batch = y_batch.to(device=self.device)
                # Reset gradient
                optimizer.zero_grad()
                # Evaluate model
                y_hat = self.model(X_batch)
                # Compute loss
                assert y_hat.shape == y_batch.shape, 'Internal Error'
                loss = loss_function(y_hat, y_batch)
                # Backpropagate gradient
                loss.backward()
                # Perform one gradient descent step
                optimizer.step()
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
                # Update training error
                loss_train   += loss.item()
                loss_train_n += 1.0
            # Record train loss
            loss_train /= loss_train_n
            hist_train.append(loss_train)

            # Get validation loss
            self.model.eval()
            if dataset_train is dataset_val:
                loss_val = loss_train
            else:
                with torch.no_grad():
                    loss_val = 0.0
                    for _, data in enumerate(loader_val, 0):
                        # Get X (inputs) and y (targets)
                        X_batch, y_batch = data
                        # Push to cuda if required
                        if self.device is not None:
                            X_batch = X_batch.to(device=self.device)
                            y_batch = y_batch.to(device=self.device)
                        y_hat = self.model(X_batch)
                        assert y_hat.shape == y_batch.shape, 'Internal Error'
                        loss_val += y_batch.size(0)*loss_function(y_hat, y_batch).item()
                    loss_val /= len(loader_val.dataset)
                    # Record validation loss
                    hist_val.append(loss_val)

            # If verbose print validation loss
            if self.verbose:
                if es.val_loss_min is not None:
                    print(f'Epoch {_epoch}: Loss train / val: {loss_train:.5f} / {loss_val:.5f} (EarlyStopping: {es.counter} / {es.patience}; Best: {es.val_loss_min:.5f})')
                else:
                    print(f'Epoch {_epoch}: Loss train / val: {loss_train:.5f} / {loss_val:.5f} (EarlyStopping: {es.counter} / {es.patience})')
                sys.stdout.flush()
            # Check EarlyStopping
            if es(loss_val, self.model):
                break

        if self.verbose and _epoch == self.epochs-1:
            print(f'Maximum number of epochs reached')
            sys.stdout.flush()

        self.model.load_state_dict(es.model_state)
        self.model.eval()

        if dataset_train is dataset_val:
            return {'train_loss': hist_train}
        else:
            return {'train_loss': hist_train, 'val_loss': hist_val}

    def _get_optimizer(self):
        # Get optimizer specified by the user
        optimizer = self.optimizer(self.model.parameters(), **self.optimargs)
        scheduler = None
        # Update learning rate scheduler
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.schedargs)
        return optimizer, scheduler

    def get_model(self):
        return self.model

    def predict(self, *args, device=None, **kwargs):
        if device is None:
            device = self.device
        return self.model.predict(*args, device=device, **kwargs)

## Lamb Optimizer
## ----------------------------------------------------------------------------

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
        Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes.
        _Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 adam=False,
                 min_trust=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if min_trust and not 0.0 <= min_trust < 1.0:
            raise ValueError(f"Minimum trust range from 0 to 1: {min_trust}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.min_trust = min_trust
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    err_msg = "Lamb does not support sparse gradients, " + \
                        "consider SparseAdam instad."
                    raise RuntimeError(err_msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                # v_t
                # exp_avg_sq.mul_(beta2).addcmul_((1 - beta2) * grad *
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    "lr"
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(group["weight_decay"], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                if self.min_trust:
                    trust_ratio = max(trust_ratio, self.min_trust)
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio * adam_step)

        return loss
