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

from torch_geometric.data   import InMemoryDataset
from torch_geometric.data   import Data       as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

from .features_coding    import NumOxidations, NumGeometries
from .features_datatypes import CoordinationFeatures

from .model_config    import DefaultCoordinationNetConfig
from .model_data      import Batch

## ----------------------------------------------------------------------------

class BatchCENGraph(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains ligand information

        # Determine the number of ligands (m) and
        # the number sites (p)
        m, p = self.__compute_m_and_p__(cofe_list)

        # The summation matrix allows to reduce a batch of ligands
        # to a batch of sites across several materials
        self.summation = self.__compute_s__(cofe_list, m, p)

        # Extract data from features objects
        self.graphs    = self.__compute_graphs__(cofe_list)

    def __compute_graphs__(self, cofe_list : list[CoordinationFeatures]) -> None:
        r = []
        for features in cofe_list:
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
                        'distance'  : torch.tensor(nb['distance'], dtype=torch.float),
                    }
                    e = [[], []]
                    for j, k in enumerate(nb['ligand_indices']):
                        # From          ; To
                        e[0].append(  0); e[1].append(j+2)
                        e[0].append(j+2); e[1].append(  0)
                        e[0].append(  1); e[1].append(j+2)
                        e[0].append(j+2); e[1].append(  1)

                    r.append(
                        GraphData(x=x, edge_index=torch.tensor(e, dtype=torch.long)))

        return r                    

    @classmethod
    def __compute_m_and_p__(cls, cofe_list : list[CoordinationFeatures]) -> tuple[int, int]:
        m = 0
        p = 0
        # Loop over materials
        for features in cofe_list:
            # Count the number of sites
            p += len(features.sites.elements)
            # Count number of CE pairs
            m += len(features.ce_neighbors)

        return m, p

    @classmethod
    def __compute_s__(cls, cofe_list : list[CoordinationFeatures], m : int, p : int) -> torch.Tensor:
        # Construct summation matrix
        S = torch.zeros(m, p)
        # CE index (rows)
        i = 0
        # Material index (columns)
        j = 0
        # Loop over materials
        for features in cofe_list:
            # Loop over CE neighbor pairs
            for nb in features.ce_neighbors:
                # For all ligands of this CE pair...
                l = len(nb['ligand_indices'])
                if l > 0:
                    # ... set the summation matrix to 1 and normalize later
                    S[i, j] = 1.0 / len(features.ce_neighbors); i += 1
            # Increment material index
            j += 1

        return S

## ----------------------------------------------------------------------------

class GraphCoordinationFeaturesLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs) -> None:

        if 'collate_fn' in kwargs:
            raise TypeError(f'{self.__class__}.__init__() got an unexpected keyword argument \'collate_fn\'')

        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        x = [ item[0] for item in batch ]
        y = [ item[1] for item in batch ]

        return BatchCENGraph(x), torch.utils.data.default_collate(y)
