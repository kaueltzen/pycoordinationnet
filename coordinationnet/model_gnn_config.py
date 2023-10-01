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

_graph_config = {
    'layers'        : None,
    'dim_element'   : None,
    'dim_oxidation' : None,
    'dim_geometry'  : None,
    'dim_csm'       : None,
    'dim_distance'  : None,
    'dim_angle'     : None,
    'bins_csm'      : None,
    'bins_distance' : None,
    'bins_angle'    : None,
    'distances'     : False,
    'angles'        : False,
}

## ----------------------------------------------------------------------------

GraphCoordinationNetConfig = ModelConfig(_graph_config)

DefaultGraphCoordinationNetConfig = ModelConfig(_graph_config)(
    layers        = [512, 128, 1],
    dim_element   = 200,
    dim_oxidation = 10,
    dim_geometry  = 10,
    dim_csm       = 128,
    dim_distance  = 128,
    dim_angle     = 128,
    bins_csm      = 20,
    bins_distance = 20,
    bins_angle    = 20,
    distances     = True,
    angles        = True)
