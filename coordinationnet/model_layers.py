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

import math
import numpy  as np
import pandas as pd
import torch
import os

from .features_coding import NumElements

## ----------------------------------------------------------------------------

class TorchStandardScaler(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        # Always use requires_grad=False, since we do not want to update
        # parameters during training. However, we must store mean and standard
        # deviations in a Parameter module, so that both get automatically
        # pushed to GPU when required
        self.mean = torch.nn.Parameter(torch.tensor(dim*[0.0]), requires_grad=False)
        self.std  = torch.nn.Parameter(torch.tensor(dim*[1.0]), requires_grad=False)

    def fit(self, x):
        self.mean = torch.nn.Parameter(x.mean(0, keepdim = False                  )       , requires_grad=False)
        self.std  = torch.nn.Parameter(x.std (0, keepdim = False, unbiased = False) + 1e-8, requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x*self.std + self.mean

## ----------------------------------------------------------------------------

class ModelDense(torch.nn.Module):
    def __init__(self, ks, skip_connections=True, dropout=False, layernorm=False, batchnorm=False, batchnorm_momentum=0.1, batchnorm_out=False, activation=torch.nn.ELU(), activation_out=None, seed=None):
        super().__init__()
        if len(ks) < 2:
            raise ValueError("invalid argument: ks must have at least two values for input and output") 
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.activation       = activation
        self.activation_out   = activation_out
        self.skip_connections = skip_connections
        self.linear           = torch.nn.ModuleList([])
        self.linear_skip      = torch.nn.ModuleList([])
        self.layernorm        = torch.nn.ModuleList([])
        self.batchnorm        = torch.nn.ModuleList([])
        self.batchnorm_skip   = torch.nn.ModuleList([])
        self.batchnorm_out    = None
        self.dropout          = torch.nn.ModuleList([])
        for i in range(0, len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))
            if type(dropout) == float:
                self.dropout.append(torch.nn.Dropout(dropout))
        for i in range(0, len(ks)-2):
            if layernorm:
                self.layernorm.append(torch.nn.LayerNorm(ks[i+1]))
            if batchnorm:
                self.batchnorm.append(torch.nn.BatchNorm1d(ks[i+1], momentum=batchnorm_momentum))
            if skip_connections:
                self.linear_skip.append(torch.nn.Linear(ks[i], ks[i+1], bias=False))
            if batchnorm and skip_connections:
                self.batchnorm_skip.append(torch.nn.BatchNorm1d(ks[i+1], momentum=batchnorm_momentum))
        # Optional: batch norm for output layer
        if batchnorm_out:
            self.batchnorm_out = torch.nn.BatchNorm1d(ks[-1], momentum=batchnorm_momentum)

    def block(self, x, i):
        # First, apply dropout
        if len(self.dropout) > 0:
            x = self.dropout[i](x)
        # Apply linear layer
        y = self.linear[i](x)
        # Normalize output
        if len(self.layernorm) > 0:
            y = self.layernorm[i](y)
        if len(self.batchnorm) > 0:
            y = self.batchnorm[i](y)
        # Apply activation
        y = self.activation(y)
        # Apply skip-connections (ResNet)
        if len(self.linear_skip) > 0:
            x = self.linear_skip[i](x)
            if len(self.batchnorm_skip) > 0:
                x = self.batchnorm_skip[i](x)
        if type(self.skip_connections) == int  and i % self.skip_connections == 0:
            y = (y + x)/2.0
        if type(self.skip_connections) == bool and self.skip_connections:
            y = (y + x)/2.0

        return y

    def block_final(self, x):
        # First, apply dropout
        if len(self.dropout) > 0:
            x = self.dropout[-1](x)
        # Apply final layer if available
        if len(self.linear) >= 1:
            x = self.linear[-1](x)
        # Normalize output
        if self.batchnorm_out is not None:
            x = self.batchnorm_out(x)
        # Apply output activation if available
        if self.activation_out is not None:
            x = self.activation_out(x)

        return x

    def forward(self, x):
        # Apply innear layers
        for i in range(len(self.linear)-1):
            x = self.block(x, i)
        # Apply final layer if available
        x = self.block_final(x)

        return x

## ----------------------------------------------------------------------------

class RBFLayer(torch.nn.Module):
    def __init__(self, vmin : float, vmax : float, bins : int = 40, gamma : float = None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )
        self.gamma = bins/math.fabs(vmax-vmin) if gamma is None else gamma
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = -self.gamma * (x.unsqueeze(1) - self.centers) ** 2
        return torch.exp(x)

## ----------------------------------------------------------------------------

class AngleLayer(torch.nn.Module):
    def __init__(self, edim, layers, **kwargs):
        super().__init__()

        self.dense = ModelDense([edim+3] + layers + [edim], **kwargs)

    def forward(self, x, x_distances, x_angles):
        x_angles = x_angles / 180 * 2*torch.pi
        x_angles = torch.cat((torch.sin(x_angles), torch.cos(x_angles)), dim=1)
        return self.dense(torch.cat((x, x_distances, x_angles), dim=1))

## ----------------------------------------------------------------------------

class PaddedEmbedder(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings+1, embedding_dim, **kwargs)
        self.weight.data[num_embeddings-1][:] = 0

## ----------------------------------------------------------------------------

class ElementEmbedder(torch.nn.Module):
    def __init__(self, edim, from_pretrained=True, freeze=True):
        super().__init__()
        if from_pretrained:
            currdir   = os.path.dirname(os.path.realpath(__file__))
            mat2vec   = os.path.join(currdir, 'model_layers_mat2vec.csv')
            embedding = pd.read_csv(mat2vec, index_col=0).values
            feat_size = embedding.shape[-1]
            embedding = np.concatenate([embedding, np.zeros((1, feat_size))])
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
            self.embedding = torch.nn.Embedding.from_pretrained(embedding, freeze=freeze)
            if edim == 200:
                self.linear = torch.nn.Identity()
            else:
                self.linear = torch.nn.Linear(feat_size, edim, bias=False)
        else:
            self.embedding = torch.nn.Embedding(NumElements+1, edim)
            self.linear    = torch.nn.Identity()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x
