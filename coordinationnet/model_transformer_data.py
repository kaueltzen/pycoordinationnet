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

from __future__ import annotations

import torch

from copy import copy
from tqdm import tqdm

from .features_datatypes import CoordinationFeatures
from .features_coding    import NumElements, NumGeometries, NumOxidations

from .model_config       import CoordinationNetConfig
from .model_data         import CoordinationFeaturesData

## ----------------------------------------------------------------------------

class Batch():

    # This function is used by the estimator to push
    # data to GPU
    def to(self, device=None):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, 'to'):
                 result.__setattr__(attr, value.to(device=device))
        return result

    # This function will be called by the pytorch DataLoader
    # after collate_fn has assembled the batch
    def pin_memory(self):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, 'pin_memory'):
                 result.__setattr__(attr, value.pin_memory())
        return result

## ----------------------------------------------------------------------------

# Composition batch data structure:
#
# m : number of materials (number of rows)
#
#       CLS ElementId_Site#1 ElementId_Site#2 ElementId_Site#3
# Mat#1 X   X                X                <padding>
# Mat#2 X   X                X                X
# Mat#3 X   X                <padding>        <padding>
# ...

## ----------------------------------------------------------------------------

class BatchComposition(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains all elements of a site and
        # information on the oxidation state

        # Number of materials
        m = len(cofe_list)

        self.elements = len(cofe_list)*[None]
        self.sizes    = len(cofe_list)*[None]

        for i, features in enumerate(cofe_list):
            self.elements[i] = torch.tensor(features.sites.elements, dtype=torch.int)
            self.sizes   [i] = len(features.sites.elements)

        # Allocate batch data
        self.cls      = torch.zeros((m, 1), dtype=torch.int)
        self.elements = torch.nn.utils.rnn.pad_sequence(self.elements, batch_first = True, padding_value = NumElements)
        self.sizes    = torch.tensor(self.sizes, dtype=torch.float)

## ----------------------------------------------------------------------------

# Sites batch data structure that combines ElementIds and OxidationIds:
#
# m : number of materials (number of rows)
#
#       CLS ElementId+OxidId_Site#1 ElementId+OxidId_Site#2 ElementId+OxidId_Site#3
# Mat#1 X   X                       X                       <padding>
# Mat#2 X   X                       X                       X
# Mat#3 X   X                       <padding>               <padding>
# ...

## ----------------------------------------------------------------------------

class BatchSites(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures], oxidation = True, ces = True) -> None:
        super().__init__()

        # This batch contains all elements of a site and
        # information on the oxidation state

        # Number of materials
        m = len(cofe_list)

        self.elements   = len(cofe_list)*[None]
        self.oxidations = len(cofe_list)*[None] if oxidation else None
        self.ces        = len(cofe_list)*[None] if ces       else None

        for i, features in enumerate(cofe_list):
            self.elements[i] = torch.tensor(features.sites.elements, dtype=torch.int)
            # Add optional features
            if oxidation:
                self.oxidations[i] = torch.tensor(features.sites.oxidations, dtype=torch.int)
            if ces:
                self.ces[i] = self.__get_ces__(features)

        # Allocate batch data
        self.cls        = torch.zeros((m, 1), dtype=torch.int)
        self.elements   = torch.nn.utils.rnn.pad_sequence(self.elements  , batch_first = True, padding_value = NumElements)
        self.oxidations = torch.nn.utils.rnn.pad_sequence(self.oxidations, batch_first = True, padding_value = NumOxidations) if oxidation else None
        self.ces        = torch.nn.utils.rnn.pad_sequence(self.ces       , batch_first = True, padding_value = NumGeometries) if ces       else None
        # Generate mask: True = {mask value}, False = {don't mask value}
        self.mask       = self.elements == NumElements
        self.mask       = torch.cat((torch.tensor([m*[False]]).T, self.mask), dim=1)

    def __get_ces__(self, features):
        # Each site may have multiple CEs, but in almost all cases a site fits only one CE.
        # Some sites (anions) do not have any site information, where we use the value
        # `NumGeometries`. Note that this value is also used for padding, but the mask
        # prevents the transformer to attend to padded values
        ces = torch.tensor(len(features.sites.elements)*[NumGeometries], dtype=torch.int)
        for ce in features.ces:
            # Get site index
            j = ce['site']
            # Consider only the first CE symbol
            ces[j] = ce['ce_symbols'][0]
        return ces

## ----------------------------------------------------------------------------

# Sites features batch data structure:
#
# m : total number of sites across all materials (number of rows)
#
#              CLS ElementId Oxidation
# Mat#1 Site#1 X   X         X
# Mat#1 Site#2 X   X         X
# ...
# Mat#2 Site#1 X   X         X
# Mat#2 Site#2 X   X         X
# ...
#
# After transforming the data, we take the first column <CLS> and
# apply the summation matrix, leading to:
#
#       CLS
# Mat#1 X
# Mat#2 X
# ...

## ----------------------------------------------------------------------------

class BatchSiteFeatures(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains all elements of a site and
        # information on the oxidation state

        # Determine the number of sites, which the corresponds to the number of
        # items in the batch
        m = self.__compute_m__(cofe_list)

        # Allocate batch data
        self.cls        = torch.zeros((m, 1), dtype=torch.int)
        self.elements   = torch.zeros((m, 1), dtype=torch.int)
        self.oxidations = torch.zeros((m, 1), dtype=torch.int)
        self.summation  = self.__compute_s__(cofe_list, m)

        # Extract data from features objects
        self.__init_data__(cofe_list)

    def __init_data__(self, cofe_list : list[CoordinationFeatures]) -> None:
        offset = 0
        for features in cofe_list:
            n = len(features.sites.elements)

            self.elements  [offset:(offset+n), 0] = torch.tensor(features.sites.elements  , dtype=torch.int)
            self.oxidations[offset:(offset+n), 0] = torch.tensor(features.sites.oxidations, dtype=torch.int)

            offset += n

    @classmethod
    def __compute_m__(cls, cofe_list : list[CoordinationFeatures]) -> int:
        m = 0
        for features in cofe_list:
            m += len(features.sites.elements)
        return m

    @classmethod
    def __compute_s__(cls, cofe_list : list[CoordinationFeatures], m : int) -> torch.Tensor:
        # Construct summation matrix
        n = len(cofe_list)
        S = torch.zeros(m, n)
        # Site index (rows)
        i = 0
        # Loop over materials (columns)
        for j, features in enumerate(cofe_list):
            l = len(features.sites.elements)
            if l > 0:
                S[i:(i+l), j] = 1.0 / l
                i += l
        return S

## ----------------------------------------------------------------------------

# Coordination environment per site batch:
#
# Coordination environments (CEs) are available for all cations. Each cation might
# have one or more coordination environments. Sites that do not have a coordination
# environment (anions) are padded
#
# m : total number of CEs across all materials (number of rows)
# p : total number of sites across all materials (target dimension after summation)
#
#            CLS CeSymbol
# Mat#1 CE#1 X   X
# Mat#1 CE#2 X   X
# ...
# Mat#2 CE#1 X   X
# Mat#2 CE#2 X   X
# ...
#
# After transforming the data, we take the first column <CLS> and
# apply the summation matrix, leading to:
#
#              CLS
# Mat#1 Site#1 X
# Mat#1 Site#2 X
# ...
# Mat#2 Site#1 X
# Mat#2 Site#2 X
# ...

## ----------------------------------------------------------------------------

class BatchCeSites(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains for each sites the coordination
        # environments

        # Determine the number of coordination environments (m) and
        # the number of sites (p)
        m, p = self.__compute_m_and_p__(cofe_list)

        # Allocate batch data
        self.cls        = torch.zeros((m, 1), dtype=torch.int)
        self.ce_symbols = torch.zeros((m, 1), dtype=torch.int)
        self.csms       = torch.zeros((m, 1), dtype=torch.float)
        # The summation matrix allows to reduce a batch of coordination
        # environments to a batch of sites across several materials
        self.__compute_s_and_index__(cofe_list, m, p)

        # Extract data from features objects
        self.__init_data__(cofe_list)

    def __init_data__(self, cofe_list : list[CoordinationFeatures]) -> None:
        # Initialize ce_symbols to padding value
        self.ce_symbols[:,0] = NumGeometries
        # CE index
        i = 0
        # Loop over materials
        for features in cofe_list:
            # Map site indices to coordination environments
            ce_symbols = len(features.sites.elements)*[None]
            csms       = len(features.sites.elements)*[None]
            # Loop over coordination environments
            for ces in features.ces:
                i_site = ces['site']
                ce_symbols[i_site] = ces['ce_symbols']
                csms      [i_site] = ces['csms']
            # Fill results into our data vectors
            for i_site, _ in enumerate(features.sites.elements):
                if ce_symbols[i_site] is None:
                    # This site does not have a coordination environment, we
                    # simply use the padding value here, which has already
                    # been filled in; Increment CE index
                    i += 1
                else:
                    # Insert all ce_symbols
                    for k, ce_symbol in enumerate(ce_symbols[i_site]):
                        self.ce_symbols[i+k] = ce_symbol
                    # Insert all csms
                    for k, csm in enumerate(csms[i_site]):
                        self.csms      [i+k] = csm
                    # Increment CE index
                    i += len(ce_symbols[i_site])

    @classmethod
    def __compute_m_and_p__(cls, cofe_list : list[CoordinationFeatures]) -> tuple[int, int]:
        # Number of coordination environments (including padded CEs for anions)
        m = 0
        # Number of sites
        p = 0
        # Loop over materials
        for features in cofe_list:
            # Count number of sites
            p += len(features.sites.elements)
            # Map site indices to coordination environments
            ce_symbols = len(features.sites.elements)*[None]
            # Loop over coordination environments
            for ces in features.ces:
                i_site = ces['site']
                ce_symbols[i_site] = ces['ce_symbols']
            # Fill results into our data vectors
            for i_site, _ in enumerate(features.sites.elements):
                if ce_symbols[i_site] is None:
                    m += 1
                else:
                    m += len(ce_symbols[i_site])
        return m, p

    def __compute_s_and_index__(self, cofe_list : list[CoordinationFeatures], m : int, p : int) -> torch.Tensor:
        # Construct an index that maps (mat,site) -> {index of final output}
        index = {}
        # Construct summation matrix
        S = torch.zeros(m, p)
        # CE index
        i = 0
        # Site index
        j = 0
        # Loop over materials
        for i_mat, features in enumerate(cofe_list):
            # Map site indices to list of coordination environments for the given site
            ce_fractions = len(features.sites.elements)*[None]
            # Loop over coordination environments
            for ces in features.ces:
                i_site = ces['site']
                ce_fractions[i_site] = ces['ce_fractions']
            # Fill results into our data vectors
            for i_site, _ in enumerate(features.sites.elements):
                if ce_fractions[i_site] is None:
                    # This site does not have a coordination environment,
                    # we just map the padding value to the site
                    S[i, j] = 1
                    # Increment CE index
                    i += 1
                else:
                    # Insert all ce_fractions
                    for k, ce_fraction in enumerate(ce_fractions[i_site]):
                        S[i+k, j] = ce_fraction
                    # Increment CE index
                    i += len(ce_fractions[i_site])
                # Save index
                index[ (i_mat, i_site) ] = j
                # Increment site index
                j += 1

        self.summation = S
        self.index     = index

## ----------------------------------------------------------------------------

# Ligands per sites batch:
#
# Coordination environments (CEs) are connected through ligands. Depending
# on the type of connection, one ore more ligands are involved.
#
# m : total number of ligands across all CE pairs and materials (number of rows)
# p : number of sites (target dimension after summation)
#
#                CLS Element1 Element2 Ligand Angle
# Mat#1 Ligand#1 X   X        X        X      X
# Mat#1 Ligand#2 X   X        X        X      X
# ...
# Mat#2 Ligand#1 X   X        X        X      X
# Mat#2 Ligand#2 X   X        X        X      X
# ...
#
# After transforming the data, we take the first column <CLS> and
# apply the summation matrix (m x p), leading to:
#
#             CLS
# Mat#1 Site1 X
# Mat#1 Site2 X
# ...
# Mat#2 Site1 X
# Mat#2 Site2 X

## ----------------------------------------------------------------------------

class BatchLigandSites(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains ligand information

        # Determine the number of ligands (m) and
        # the number sites (p)
        m, p = self.__compute_m_and_p__(cofe_list)

        # Allocate batch data
        self.cls      = torch.zeros((m, 1), dtype=torch.int)
        self.elements = torch.zeros((m, 2), dtype=torch.int)
        self.ligelem  = torch.zeros((m, 1), dtype=torch.int)
        self.ligoxid  = torch.zeros((m, 1), dtype=torch.int)
        self.angles   = torch.zeros((m, 1), dtype=torch.float)
        # The summation matrix allows to reduce a batch of ligands
        # to a batch of sites across several materials
        self.summation  = self.__compute_s__(cofe_list, m, p)

        # Extract data from features objects
        self.__init_data__(cofe_list)

    def __init_data__(self, cofe_list : list[CoordinationFeatures]) -> None:
        i = 0
        for features in cofe_list:
            for nb in features.ce_neighbors:
                l = len(nb['ligand_indices'])
                if l > 0:
                    self.elements[i:(i+l), 0] = features.sites.elements[nb['site'   ]]
                    self.elements[i:(i+l), 1] = features.sites.elements[nb['site_to']]
                    for j, k in enumerate(nb['ligand_indices']):
                        self.ligelem[i+j, 0] = features.sites.elements  [k]
                        self.ligoxid[i+j, 0] = features.sites.oxidations[k]
                        self.angles [i:j, 0] = nb['angles'][j]
                    i += l

    @classmethod
    def __compute_m_and_p__(cls, cofe_list : list[CoordinationFeatures]) -> tuple[int, int]:
        m = 0
        p = 0
        # Loop over materials
        for features in cofe_list:
            # Count the number of sites
            p += len(features.sites.elements)
            # Loop over CE neighbor pairs
            for nb in features.ce_neighbors:
                # Count the number of ligands
                m += len(nb['ligand_indices'])
        return m, p

    @classmethod
    def __compute_s__(cls, cofe_list : list[CoordinationFeatures], m : int, p : int) -> torch.Tensor:
        # Construct summation matrix
        S = torch.zeros(m, p)
        # Ligand index (rows)
        i = 0
        # Site index (columns)
        j = 0
        # Loop over materials
        for features in cofe_list:
            # Loop over CE neighbor pairs
            for nb in features.ce_neighbors:
                # For all ligands of this CE pair...
                l = len(nb['ligand_indices'])
                if l > 0:
                    # ... set the summation matrix to 1 and normalize later
                    for ligand in nb['ligand_indices']:
                        S[i, j+ligand] = 1; i += 1
            # Increment site index
            j += len(features.sites.elements)

        # Compute normalization vector (normalize columns)
        z = S.sum(dim=0)
        # Some entries are zero (i.e. some sites do not have ligand information),
        # replace by 1
        z[z == 0] = 1

        return S / z

## ----------------------------------------------------------------------------

# Ligands batch:
#
# Coordination environments (CEs) are connected through ligands. Depending
# on the type of connection, one ore more ligands are involved.
#
# m : total number of ligands across all CE pairs and materials (number of rows)
# p : number of CE-pairs (target dimension after summation)
#
#                CLS Element1 Element2 Ligand Angle
# Mat#1 Ligand#1 X   X        X        X      X
# Mat#1 Ligand#2 X   X        X        X      X
# ...
# Mat#2 Ligand#1 X   X        X        X      X
# Mat#2 Ligand#2 X   X        X        X      X
# ...
#
# After transforming the data, we take the first column <CLS> and
# apply the summation matrix (m x p), leading to:
#
#           CLS
# CE-Pair#1 X
# CE-Pair#2 X
# ...

## ----------------------------------------------------------------------------

class BatchLigands(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures]) -> None:
        super().__init__()

        # This batch contains ligand information

        # Determine the number of ligands (m) and
        # the number of ce pairs (p)
        m, p = self.__compute_m_and_p__(cofe_list)

        # Allocate batch data
        self.cls      = torch.zeros((m, 1), dtype=torch.int)
        self.elements = torch.zeros((m, 2), dtype=torch.int)
        self.ligands  = torch.zeros((m, 1), dtype=torch.int)
        self.angles   = torch.zeros((m, 1), dtype=torch.float)
        # The summation matrix allows to reduce a batch of ligands
        # to a batch of sites across several materials
        self.summation  = self.__compute_s__(cofe_list, m, p)

        # Extract data from features objects
        self.__init_data__(cofe_list)

    def __init_data__(self, cofe_list : list[CoordinationFeatures]) -> None:
        i = 0
        for features in cofe_list:
            for nb in features.ce_neighbors:
                l = len(nb['ligand_indices'])
                if l > 0:
                    self.elements[i:(i+l), 0] = nb['site']
                    self.elements[i:(i+l), 1] = nb['site_to']
                    self.ligands [i:(i+l), 0] = torch.tensor(nb['ligand_indices'], dtype=torch.int)
                    self.angles  [i:(i+l), 0] = torch.tensor(nb['angles']        , dtype=torch.float)
                    i += l

    @classmethod
    def __compute_m_and_p__(cls, cofe_list : list[CoordinationFeatures]) -> tuple[int, int]:
        m = 0
        p = 0
        # Loop over materials
        for features in cofe_list:
            # Count the number of pairs
            p += len(features.ce_neighbors)
            # Loop over CE neighbor pairs
            for nb in features.ce_neighbors:
                # Count the number of ligands
                m += len(nb['ligand_indices'])
        return m, p

    @classmethod
    def __compute_s__(cls, cofe_list : list[CoordinationFeatures], m : int, p : int) -> torch.Tensor:
        # Construct summation matrix
        S = torch.zeros(m, p)
        # Ligand index (rows)
        i = 0
        # CE-pair index (columns)
        j = 0
        # Loop over materials
        for features in cofe_list:
            # Loop over CE neighbor pairs
            for nb in features.ce_neighbors:
                # For all ligands of this CE pair...
                l = len(nb['ligand_indices'])
                if l > 0:
                    # ... set the summation matrix to 1/l
                    S[i:(i+l), j] = 1/l
                    # Increment ligand index
                    i += l
                # Increment CE-pair index
                j += 1

        return S

## ----------------------------------------------------------------------------

# Batch containing pairs of neighboring coordination environments (CEs) with
# information on how they are connected.
#
# m: Toal number of CE pairs (number of rows)
#
#                CLS Element1 Element2 Distance Connectivity
# Mat#1 CePair#1 X   X        X        X        X
# Mat#1 CePair#2 X   X        X        X        X
# ...
# Mat#2 CePair#1 X   X        X        X        X
# Mat#2 CePair#2 X   X        X        X        X
# ...
#
# After transforming the data, we take the first column <CLS> and
# apply the summation matrix, leading to:
#
#       CLS
# Mat#1 X
# Mat#2 X
# ...

## ----------------------------------------------------------------------------

class BatchCeNeighbors(Batch):

    def __init__(self, cofe_list : list[CoordinationFeatures], site_features_ces : BatchCeSites) -> None:
        super().__init__()

        # This is the final batch containing information
        # on the connectivity between two coordination
        # environments

        # Determine the number of CE pairs, which also corresponds to the number of
        # items in the batch
        m = self.__compute_m__(cofe_list)

        # Allocate batch data
        self.cls          = torch.zeros((m, 1), dtype=torch.int)
        self.elements     = torch.zeros((m, 2), dtype=torch.int)
        self.oxidation    = torch.zeros((m, 2), dtype=torch.int)
        self.distances    = torch.zeros((m, 1), dtype=torch.float)
        self.connectivity = torch.zeros((m, 1), dtype=torch.int)
        self.summation    = self.__compute_s__(cofe_list, m)

        # Extract data from features objects
        self.__init_data__(cofe_list, site_features_ces)

    def __init_data__(self, cofe_list : list[CoordinationFeatures], site_features_ces : BatchCeSites) -> None:
        offset = 0
        for i_mat, features in enumerate(cofe_list):
            for i, nb in enumerate(features.ce_neighbors):
                self.elements    [(offset+i), 0] = features.sites.elements[nb['site'   ]]
                self.elements    [(offset+i), 1] = features.sites.elements[nb['site_to']]
                self.oxidation   [(offset+i), 0] = features.sites.elements[nb['site'   ]]
                self.oxidation   [(offset+i), 1] = features.sites.elements[nb['site_to']]
                self.connectivity[(offset+i), 0] = nb['connectivity']
                self.distances   [(offset+i), 0] = nb['distance']

            offset += len(features.ce_neighbors)

    @classmethod
    def __compute_m__(cls, cofe_list : list[CoordinationFeatures]) -> int:
        m = 0
        for features in cofe_list:
            m += len(features.ce_neighbors)
        return m

    @classmethod
    def __compute_s__(cls, cofe_list : list[CoordinationFeatures], m : int) -> torch.Tensor:
        # Number of materials
        n = len(cofe_list)
        # Construct summation matrix
        S = torch.zeros(m, n)
        # Site index (rows)
        i = 0
        # Loop over materials (columns)
        for j, features in enumerate(cofe_list):
            l = len(features.ce_neighbors)
            q = len(features.sites.elements)
            if l == 0:
                continue
            # Count the number of CE-pairs per site
            c = torch.zeros(q)
            # Normalization constant for each CE-pair
            z = torch.zeros(l)
            for nb in features.ce_neighbors:
                c[nb['site'   ]] += 1.0
                c[nb['site_to']] += 1.0
            for k, nb in enumerate(features.ce_neighbors):
                z[k] = 1.0/c[nb['site']] + 1.0/c[nb['site_to']]
            # Assign weights to all ce-pairs
            S[i:(i+l), j] = z / z.sum() / q
            i += l
        return S

## ----------------------------------------------------------------------------

class BatchCoordinationFeatures(Batch):
    def __init__(self, cofe_list : list[CoordinationFeatures], model_config : CoordinationNetConfig) -> None:
        super().__init__()

        self.composition           = None
        self.sites                 = None
        self.site_features         = None
        self.site_features_ces     = None
        self.site_features_ligands = None
        self.ligands               = None
        self.ce_neighbors          = None

        if model_config['ce_neighbors'] and not model_config['site_features_ces']:
            raise ValueError('Option `ce_neighbors` requires `site_features_ces`')

        if model_config['composition']:
            self.composition = BatchComposition(cofe_list)

        if model_config['sites']:
            self.sites = BatchSites(cofe_list, oxidation = model_config['sites_oxid'], ces = model_config['sites_ces'])

        if model_config['site_features']:
            self.site_features = BatchSiteFeatures(cofe_list)

        if model_config['site_features_ces']:
            self.site_features_ces = BatchCeSites(cofe_list)

        if model_config['site_features_ligands']:
            self.site_features_ligands = BatchLigandSites(cofe_list)

        if model_config['ligands']:
            self.ligands = BatchLigands(cofe_list)

        if model_config['ce_neighbors']:
            self.ce_neighbors = BatchCeNeighbors(cofe_list, self.site_features_ces)

    def _check_config(self, model_config : dict, key):
        if key in model_config:
            return model_config[key]
        else:
            return False

## ----------------------------------------------------------------------------

class BatchedCoordinationFeaturesData(torch.utils.data.Dataset):
    def __init__(self, dataset : CoordinationFeaturesData, model_config, batch_size : int, drop_last = False, **kwargs) -> None:

        sampler = torch.utils.data.RandomSampler(range(len(dataset)))
        sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last = drop_last)

        self.dataset = []
        for batch_idx in tqdm(sampler, desc='Preparing batches...'):
            X = BatchCoordinationFeatures(
                [ dataset[i][0] for i in batch_idx ], model_config)
            y = torch.utils.data.default_collate(
                [ dataset[i][1] for i in batch_idx ])
            self.dataset.append((X, y))

    def __len__(self) -> int:
        return len(self.dataset)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        return self.dataset[index]

## ----------------------------------------------------------------------------

class CoordinationFeaturesLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset : CoordinationFeaturesData, model_config, batch_size = 1, **kwargs) -> None:
        if 'collate_fn' in kwargs:
            raise TypeError(f'{self.__class__}.__init__() got an unexpected keyword argument \'collate_fn\'')

        # Generate batches in advance to speed up computation
        dataset = BatchedCoordinationFeaturesData(dataset, model_config, batch_size)

        super().__init__(dataset, batch_size=1, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        return batch[0]
