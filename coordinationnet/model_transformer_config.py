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

from .model_config import ModelConfig

## ----------------------------------------------------------------------------

_transformer_config = {
    'composition'           : False,
    'sites'                 : False,
    'sites_oxid'            : False,
    'sites_ces'             : False,
    'site_features'         : False,
    'site_features_ces'     : False,
    'site_features_oxid'    : False,
    'site_features_csms'    : False,
    'site_features_ligands' : False,
    'ligands'               : False,
    'ce_neighbors'          : False,
}

## ----------------------------------------------------------------------------

TransformerCoordinationNetConfig = ModelConfig(_transformer_config)

DefaultTransformerCoordinationNetConfig = ModelConfig(_transformer_config)(
    site_features      = True,
    site_features_ces  = True,
    site_features_oxid = True,
    site_features_csms = True)
