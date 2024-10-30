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

import numpy as np

from coordinationnet import mp_icsd_query, mp_icsd_clean

#%% Retrieve oxides from materials project
### ---------------------------------------------------------------------------

mats = mp_icsd_query("Q0tUKnAE52sy7hVO", experimental_data = False)
mats = mp_icsd_clean(mats)
# Remove 'mp-554015' due to bug #2756
mats = np.delete(mats, 3394)
# Need to convert numpy array to list for serialization
mats = mats.tolist()

#%% Extract structures and target values, convert structures to
### coordination features (this may take a while)
### ---------------------------------------------------------------------------

from coordinationnet import CoordinationFeaturesData

structures = [  mat['structure']                  for mat in mats ]
targets    = [ [mat['formation_energy_per_atom']] for mat in mats ]

data = CoordinationFeaturesData(structures, y = targets, verbose = True)

#%% Save data
### ---------------------------------------------------------------------------

data.save('mpoxides.dill')

#%% Load data
### ---------------------------------------------------------------------------

data = CoordinationFeaturesData.load('mpoxides.dill')

# %%
