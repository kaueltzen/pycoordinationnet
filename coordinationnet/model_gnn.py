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
from torch_geometric.nn   import Sequential, GCNConv, global_mean_pool

from .features_coding import NumOxidations, NumGeometries

from .model_config    import DefaultCoordinationNetConfig
from .model_layers    import TorchStandardScaler, ModelDense, ElementEmbedder, RBFLayer, AngleLayer, PaddedEmbedder

## ----------------------------------------------------------------------------

class ModelGraphCoordinationNet(torch.nn.Module):
    def __init__(self,
        # Specify model components
        model_config = DefaultCoordinationNetConfig,
        # Transformer options
        edim = 200,
        # **kwargs contains options for dense layers
        layers = [512, 128, 1], **kwargs):

        super().__init__()

        print(f'{model_config}')

        # Dimension of the graph features
        fdim = edim + 10 + 10 + 2

        # The model config determines which components of the model
        # are active
        self.model_config      = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs    = TorchStandardScaler(layers[-1])

        # Embeddings
        self.embedding_element   = ElementEmbedder(edim, from_pretrained=True, freeze=True)
        self.embedding_oxidation = torch.nn.Embedding(NumOxidations, 10)
        self.embedding_geometry  = PaddedEmbedder(NumGeometries, 10)

        # Core graph network
        self.layers = Sequential('x, edge_index, batch', [
                (GCNConv(fdim, fdim), 'x, edge_index -> x'),
                torch.nn.ELU(inplace=True),
                (GCNConv(fdim, fdim), 'x, edge_index -> x'),
                torch.nn.ELU(inplace=True),
                (GCNConv(fdim, fdim), 'x, edge_index -> x'),
                torch.nn.ELU(inplace=True),
                (global_mean_pool, 'x, batch -> x'),
            ])

        # Final dense layer
        self.dense = ModelDense([fdim] + layers, **kwargs)

        print(f'Creating a GNN model with {self.n_parameters:,} parameters')

    def forward(self, x_input):

        # Get embeddings of various features
        x_elements   = self.embedding_element  (x_input.x['elements'  ])
        x_oxidations = self.embedding_oxidation(x_input.x['oxidations'])
        x_geometries = self.embedding_geometry (x_input.x['geometries'])
        x_geometries = self.embedding_geometry (x_input.x['geometries'])
        x_angles     = x_input.x['angles']

        # Concatenate embeddings to yield a single feature vector per node
        x = torch.cat((x_elements, x_oxidations, x_geometries, x_angles), dim=1)

        # Propagate features through graph network
        x = self.layers(x, x_input.edge_index, x_input.batch)
        # Apply final dense layer
        x = self.dense(x)

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        return { 'all': self.parameters() }
