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

import collections

## ----------------------------------------------------------------------------

class ModelConfig(collections.UserDict):
    def __init__(self, _model_config):
        super().__init__(self)
        for key, value in _model_config.items():
            super().__setitem__(key, value)

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

        return self

    def __setitem__(self, key, value):
        # Check that we do not accept any invalid
        # config options
        if key in self:
            super().__setitem__(key, value)
        else:
            raise KeyError(key)

    def __str__(self):
        result = 'Model config:\n'
        for key, value in self.items():
            result += f'-> {key:21}: {value}\n'
        return result

    def __copy__(self):
        return ModelConfig(self)

    def __deepcopy__(self, memo):
        result = ModelConfig(self)
        memo[id(self)] = result
        return result
