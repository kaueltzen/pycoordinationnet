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
import os
import pytest

from monty.serialization import loadfn

from coordinationnet.model_transformer      import ModelComposition, ModelSiteFeaturesTransformer
from coordinationnet.model_transformer_data import BatchComposition, BatchSiteFeatures

## ----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## ----------------------------------------------------------------------------

@pytest.fixture
def features_list():
    data = loadfn(os.path.join(root, 'test_features.json.gz'))

    X = [ value.encode() for _, value in data.items() ]

    return X

## ----------------------------------------------------------------------------

def test_model_composition(features_list):
    b1 = BatchComposition(features_list)
    b2 = BatchSiteFeatures(features_list)

    edim = 4
    m1 = ModelComposition(edim)
    m2 = ModelSiteFeaturesTransformer(edim, transformer=False, oxidation=False, csms=False, ligands=False)
    m2.embedding_element = m1.embedding_element

    assert not torch.any(m1(b1) - m2(b2, None, None) > 1e-4).item()
