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
from torch_geometric.nn   import Sequential, GraphConv, HeteroConv, global_mean_pool

from .features_coding import NumOxidations, NumGeometries

from .model_layers     import TorchStandardScaler, ModelDense, ElementEmbedder, RBFEmbedding, ZeroPadder, PaddedEmbedder
from .model_gnn_config import DefaultGraphCoordinationNetConfig

## ----------------------------------------------------------------------------

class ModelGraphCoordinationNet(torch.nn.Module):
    def __init__(self,
        # Specify model components
        model_config = DefaultGraphCoordinationNetConfig,
        # Options for dense layers
        layers, **kwargs):

        super().__init__()

        # Feature dimensions
        dim_element   = model_config['dim_element']
        dim_oxidation = model_config['dim_oxidation']
        dim_geometry  = model_config['dim_geometry']
        dim_csm       = model_config['dim_csm']
        dim_distance  = model_config['dim_distance']
        dim_angle     = model_config['dim_angle']

        dim_site   = dim_element + dim_oxidation
        dim_ce     = dim_element + dim_oxidation + dim_geometry + dim_csm
        dim_ligand = dim_element + dim_oxidation

        if model_config['distances']:
            dim_ce += dim_distance

        if model_config['angles']:
            dim_ligand += dim_angle

        # The model config determines which components of the model
        # are active
        self.model_config        = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs      = TorchStandardScaler(layers[-1])

        # RBF encoder
        self.rbf_csm             = RBFEmbedding(0.0, 1.0, bins=model_config['bins_csm'], edim=dim_csm)
        self.rbf_distances       = RBFEmbedding(0.0, 1.0, bins=model_config['bins_distance'], edim=dim_distance)
        self.rbf_angles          = RBFEmbedding(0.0, 1.0, bins=model_config['bins_angle'], edim=dim_angle)

        # Embeddings
        self.embedding_element   = ElementEmbedder(edim, from_pretrained=True, freeze=True)
        self.embedding_oxidation = torch.nn.Embedding(NumOxidations, 10)
        self.embedding_geometry  = PaddedEmbedder(NumGeometries, 10)

        self.activation          = torch.nn.ELU(inplace=True)

        # Core graph network
        self.layers = Sequential('x, edge_index, batch', [
                # Layer 1 -----------------------------------------------------------------------------------
                (HeteroConv({
                    ('site'  , '*', 'site'  ): GraphConv((dim_site, dim_site), dim_site  , add_self_loops=False),
                    ('ligand', '*', 'ce'    ): GraphConv((dim_ligand, dim_ce), dim_ce    , add_self_loops=True ),
                    ('ce'    , '*', 'ligand'): GraphConv((dim_ce, dim_ligand), dim_ligand, add_self_loops=True ),
                }), 'x, edge_index -> x'),
                # Apply activation
                (lambda x: { k : self.activation(v) for k, v in x.items()}, 'x -> x'),
                # Layer 2 -----------------------------------------------------------------------------------
                (HeteroConv({
                    ('site'  , '*', 'site'  ): GraphConv((dim_site, dim_site), dim_site  , add_self_loops=False),
                    ('ligand', '*', 'ce'    ): GraphConv((dim_ligand, dim_ce), dim_ce    , add_self_loops=True ),
                    ('ce'    , '*', 'ligand'): GraphConv((dim_ce, dim_ligand), dim_ligand, add_self_loops=True ),
                }), 'x, edge_index -> x'),
                # Apply activation
                (lambda x: { k : self.activation(v) for k, v in x.items()}, 'x -> x'),
                # Layer 4 -----------------------------------------------------------------------------------
                (HeteroConv({
                    ('ce', '*', 'site'  ): GraphConv((dim_ce, dim_site  ), dim_site  , add_self_loops=True, bias=False),
                }, aggr='mean'), 'x, edge_index -> x'),
                # Apply activation
                (lambda x: { k : self.activation(v) for k, v in x.items()}, 'x -> x'),
                # Output  -----------------------------------------------------------------------------------
                # Apply mean pooling
                (lambda x, batch: { k : global_mean_pool(v, batch[k]) for k, v in x.items() }, 'x, batch -> x'),
                # Extract only site features
                (lambda x: x['site'], 'x -> x')
            ])

        # Final dense layer
        self.dense = ModelDense([dim_site] + layers, **kwargs)

    def forward(self, x_input):

        x_site = torch.cat((
            self.embedding_element  (x_input['site'].x['elements'  ]),
            self.embedding_oxidation(x_input['site'].x['oxidations']),
            ), dim=1)

        x_ce = torch.cat((
            self.embedding_element  (x_input['ce'].x['elements'  ]),
            self.embedding_oxidation(x_input['ce'].x['oxidations']),
            self.embedding_geometry (x_input['ce'].x['geometries']),
            self.rbf_csm            (x_input['ce'].x['csms'      ]),
            ), dim=1)

        x_ligand = torch.cat((
            self.embedding_element  (x_input['ligand'].x['elements'  ]),
            self.embedding_oxidation(x_input['ligand'].x['oxidations']),
            ), dim=1)

        # Add optional features
        if self.model_config['distances']:
            x_ce = torch.cat((
                x_ce,
                self.rbf_distances(x_input['ce'].x['distances']),
                ), dim=1)

        if self.model_config['angles']:
            x_ligand = torch.cat((
                x_ligand,
                self.rbf_angles(x_input['ligand'].x['angles']),
                ), dim=1)

        # Concatenate embeddings to yield a single feature vector per node
        x = {
            'site': x_site, 'ce': x_ce, 'ligand': x_ligand,
        }
        # Propagate features through graph network
        x = self.layers(x, x_input.edge_index_dict, x_input.batch_dict)
        # Apply final dense layer
        x = self.dense(x)
        # Apply inverse transformation
        x = self.scaler_outputs.inverse_transform(x)

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        return { 'all': self.parameters() }
