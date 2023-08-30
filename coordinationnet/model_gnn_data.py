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

from tqdm import tqdm

from torch_geometric.data   import InMemoryDataset
from torch_geometric.data   import Data       as GraphData
from torch_geometric.data   import Batch      as GraphBatch
from torch_geometric.loader import DataLoader as GraphDataLoader

from .features_coding    import NumOxidations, NumGeometries
from .features_datatypes import CoordinationFeatures

from .model_config    import DefaultCoordinationNetConfig
from .model_data      import GenericDataset, Batch

## ----------------------------------------------------------------------------

class CENData(GenericDataset):

    def __init__(self, dataset) -> None:

        X = [ item[0] for item in dataset]
        y = [ item[1] for item in dataset]

        X = self.__compute_graphs__(X, verbose=False)

        super().__init__(X, y)

    @classmethod
    def __compute_graph__(cls, features : CoordinationFeatures) -> GraphData:
        # Some materials do not have CE pairs
        if len(features.ce_neighbors) == 0:
            x = {
                'elements'  : torch.empty((0,), dtype=torch.long),
                'oxidations': torch.empty((0,), dtype=torch.long),
                'ces'       : torch.empty((0,), dtype=torch.long),
                'csm'       : torch.empty((0,), dtype=torch.float),
                'angles'    : torch.empty((0,), dtype=torch.float),
                #'distance'  : torch.empty((0,), dtype=torch.float),
            }
            return GraphData(x=x, edge_index=torch.empty((2,0), dtype=torch.long))
        # Get CE symbols and CSMs
        site_ces = len(features.sites.elements)*[NumGeometries]
        site_csm = len(features.sites.elements)*[0.0]
        # Each site may have multiple CEs, but in almost all cases a site fits only one CE.
        # Some sites (anions) do not have any site information, where we use the value
        # `NumGeometries`. Note that this value is also used for padding, but the mask
        # prevents the transformer to attend to padded values
        for ce in features.ces:
            # Get site index
            j = ce['site']
            # Consider only the first CE symbol
            site_ces[j] = ce['ce_symbols'][0]
            site_csm[j] = ce['csms'][0]
        # List of CE pair graphs
        r = []
        # Construct CE graphs
        for nb in features.ce_neighbors:
            l = len(nb['ligand_indices'])
            if l > 0:
                # Get site indices
                idx = [ nb['site'], nb['site_to'] ] + nb['ligand_indices']
                # Construct graph
                x = {
                    'elements'  : torch.tensor([ features.sites.elements  [site] for site in idx ], dtype=torch.long),
                    'oxidations': torch.tensor([ features.sites.oxidations[site] for site in idx ], dtype=torch.long),
                    'ces'       : torch.tensor([ site_ces[site]                  for site in idx ], dtype=torch.long),
                    'csm'       : torch.tensor([ site_csm[site]                  for site in idx ], dtype=torch.float),
                    'angles'    : torch.tensor([ 0.0 + 0.0 ] + nb['angles'], dtype=torch.float),
                    #'distance'  : torch.tensor(nb['distance'], dtype=torch.float),
                }
                e = [[], []]
                for j, k in enumerate(nb['ligand_indices']):
                    # From          ; To
                    e[0].append(  0); e[1].append(j+2)
                    e[0].append(j+2); e[1].append(  0)
                    e[0].append(  1); e[1].append(j+2)
                    e[0].append(j+2); e[1].append(  1)

                r.append(GraphData(x=x, edge_index=torch.tensor(e, dtype=torch.long)))

        return InMemoryDataset.collate(r)[0]

    @classmethod
    def __compute_graphs__(cls, cofe_list : list[CoordinationFeatures], verbose = False) -> list[GraphData]:
        r = len(cofe_list)*[None]
        for i, features in tqdm(enumerate(cofe_list), desc='Computing graphs', disable=(not verbose), total=len(cofe_list)):
            r[i] = cls.__compute_graph__(features)

        return r

## ----------------------------------------------------------------------------

class GraphCoordinationFeaturesLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs) -> None:

        if 'collate_fn' in kwargs:
            raise TypeError(f'{self.__class__}.__init__() got an unexpected keyword argument \'collate_fn\'')

        dataset = CENData(dataset)

        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        x = [ item[0] for item in batch ]
        y = [ item[1] for item in batch ]

        return GraphBatch.from_data_list(x), torch.utils.data.default_collate(y)
