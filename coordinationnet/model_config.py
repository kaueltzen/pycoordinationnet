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

class ModelConfig(dict):
    def __init__(self, _model_config):
        super().__init__(self)
        for key, value in _model_config.items():
            super().__setitem__(key, value)

    def __call__(self, *args, **kwargs):
        # Check that we do not accept any invalid
        # config options
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(key)
            else:
                super().__setitem__(key, value)

        return self

    def __setitem__(self, key, value):
        if key in self:
            super().__setitem__(key, value)
        else:
            raise KeyError(key)

    def __str__(self):
        result = 'Model config:\n'
        for key, value in self.items():
            result += f'-> {key:21}: {value}\n'
        return result

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

## ----------------------------------------------------------------------------

_graph_config = {
    'distances' : False,
    'angles'    : False,
}

## ----------------------------------------------------------------------------

GraphCoordinationNetConfig = ModelConfig(_graph_config)

DefaultGraphCoordinationNetConfig = ModelConfig(_graph_config)(
    distances = True,
    angles    = True)
