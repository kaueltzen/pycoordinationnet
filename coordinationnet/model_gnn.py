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

import torch

from torch_geometric.data import Data

from .features_coding import NumOxidations, NumGeometries

from .model_config    import DefaultCoordinationNetConfig
from .model_layers    import TorchStandardScaler, ModelDense, ElementEmbedder, RBFLayer, AngleLayer

## ----------------------------------------------------------------------------

class ModelGraphCoordinationNet(torch.nn.Module):
    def __init__(self,
        # Specify model components
        model_config = DefaultCoordinationNetConfig,
        # Transformer options
        edim = 200,
        # **kwargs contains options for dense layers
        layers = [200, 512, 128, 1], **kwargs):

        super().__init__()

        print(f'{model_config}')

        # The model config determines which components of the model
        # are active
        self.model_config      = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs    = TorchStandardScaler(layers[-1])

        # Embeddings
        self.embedding_element = ElementEmbedder(edim, from_pretrained=True, freeze=False)
        self.embedding_ligands = ElementEmbedder(edim, from_pretrained=True, freeze=False)
        self.embedding_ces     = torch.nn.Embedding(NumGeometries+1, edim)

        # Final dense layer
        self.dense = ModelDense([edim] + layers, **kwargs)

        print(f'Creating a GNN model with {self.n_parameters:,} parameters')

    def forward(self, x):

        print(x)
        raise ValueError('')

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        return { 'all': self.parameters() }
