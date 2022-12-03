
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

## -----------------------------------------------------------------------------

class MyMSONable(MSONable):

    def dump(self, filename) -> None:
        return dumpfn(self.as_dict(), filename)

    @classmethod
    def load(self, filename):
        return loadfn(filename)

## -----------------------------------------------------------------------------

class FancyString():

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}('
        for i, (key, value) in enumerate(self.__dict__.items()):
            if i == 0:
                s += f'{key}={str(value)}'
            else:
                s += f', {key}={str(value)}'
        return s + ')'

    def __repr__(self) -> str:
        return str(self)

## -----------------------------------------------------------------------------

class Range(FancyString, MyMSONable):

    def __init__(self, start, stop) -> None:
        self.start = start
        self.stop  = stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

## -----------------------------------------------------------------------------

class FeatureSequence():

    def __init__(self) -> None:
        self.n = -1

    def add_item(self, site, indices) -> None:
        if site < self.n:
            # Seems that we are trying to add a feature for a site
            # that we saw a while back
            raise ValueError(f'Invalid order of site features: Trying to add site index {site} while last index was {self.n}')
        elif site == self.n:
            # Given site is still the same
            indices[site] = Range(indices[site][0], indices[site][1]+1)
        else:
            self.n += 1
            while site > self.n:
                if len(indices) > 0:
                    i_start = indices[self.n-1][1]
                    i_stop  = indices[self.n-1][1]
                else:
                    i_start = 0
                    i_stop  = 0
                indices.append(Range(i_start, i_stop))
                self.n += 1
            # Observed a new site
            if len(indices) > 0:
                i_start = indices[self.n-1][1]
                i_stop  = indices[self.n-1][1]+1
            else:
                i_start = 0
                i_stop  = 1
            indices.append(Range(i_start, i_stop))

## -----------------------------------------------------------------------------

class Base(FancyString, MyMSONable):

    def __init__(self, sites = None, oxidations = None, ions = None, elements = None, coordinates = None) -> None:
        self.sites       = sites       if sites       else []
        self.oxidations  = oxidations  if oxidations  else []
        self.ions        = ions        if ions        else []
        self.elements    = elements    if elements    else []
        self.coordinates = coordinates if coordinates else []

    def add_item(self, site, oxidation, ion, element, coordinates) -> None:
        if site != len(self.sites):
            raise ValueError(f'Invalid order of site features: isite={site}, isites={self.sites}')
        self.sites      .append(site)
        self.oxidations .append(oxidation)
        self.ions       .append(ion)
        self.elements   .append(element)
        self.coordinates.append(coordinates)

## -----------------------------------------------------------------------------

class Distances(FeatureSequence, FancyString, MyMSONable):

    def __init__(self, sites = None, sites_to = None, distances = None, indices = None) -> None:
        self.sites     = sites     if sites     else []
        self.sites_to  = sites_to  if sites_to  else []
        self.distances = distances if distances else []
        self.indices   = indices   if indices   else []

    def add_item(self, site, site_to, distance) -> None:
        self.sites    .append(site)
        self.sites_to .append(site_to)
        self.distances.append(distance)
    
    def get_site_features(self, site, base=None) -> list:
        result = []
        if site >= len(self.indices):
            return result
        for i in self.indices[site]:
            if base is None:
                r_site    = self.sites   [i]
                r_site_to = self.sites_to[i]
            else:
                r_site    = base.elements[self.sites[i]]
                r_site_to = base.elements[self.sites_to[i]]
            result.append([r_site, r_site_to, self.distances[i]])
        return result

## -----------------------------------------------------------------------------


class CoordinationEnvironments(FeatureSequence, FancyString, MyMSONable):

    def __init__(self, sites = None, ce_symbols = None, ce_fractions = None, csms = None, permutations = None, indices = None) -> None:
        self.sites        = sites        if sites        else []
        self.ce_symbols   = ce_symbols   if ce_symbols   else []
        self.ce_fractions = ce_fractions if ce_fractions else []
        self.csms         = csms         if csms         else []
        self.permutations = permutations if permutations else []
        self.indices      = indices      if indices      else []

    def add_item(self, site, ce_symbol, ce_fraction, csm, permutation) -> None:
        self.sites       .append(site)
        self.ce_symbols  .append(ce_symbol)
        self.ce_fractions.append(ce_fraction)
        self.csms        .append(csm)
        self.permutations.append(permutation)

    def get_site_features(self, site) -> list:
        result = []
        if site >= len(self.indices):
            return result
        for i in self.indices[site]:
            result.append({
                'ce_symbol'   : self.ce_symbols  [i],
                'ce_fraction' : self.ce_fractions[i],
                'csm'         : self.csms        [i],
                'permutation' : self.permutations[i]
            })
        return result

## -----------------------------------------------------------------------------

class Angles(FeatureSequence, FancyString, MyMSONable):

    def __init__(self, sites = None, sites_to = None, ligands = None, angles = None, indices = None) -> None:
        self.sites    = sites    if sites    else []
        self.sites_to = sites_to if sites_to else []
        self.ligands  = ligands  if ligands  else []
        self.angles   = angles   if angles   else []
        self.indices  = indices  if indices  else []

    def add_item(self, site, site_to, ligands, angles) -> None:
        self.sites    .append(site)
        self.sites_to .append(site_to)
        self.ligands  .append(ligands)
        self.angles   .append(angles)

    def get_site_features(self, site, base=None) -> list:
        result = []
        if site >= len(self.indices):
            return result
        for i in self.indices[site]:
            if base is None:
                r_site    = self.sites   [i]
                r_site_to = self.sites_to[i]
                ligands   = self.ligands [i]
            else:
                r_site    = base.elements[self.sites   [i]]
                r_site_to = base.elements[self.sites_to[i]]
                ligands   = [ base.elements[ligand] for ligand in self.ligands[i] ]
            result.append([r_site, r_site_to, ligands, self.angles[i]])
        return result

## -----------------------------------------------------------------------------

class CeAngles(FancyString, MyMSONable):

    def __init__(self, isolated = None, corner = None, edge = None, face = None) -> None:
        self.isolated = isolated if isolated else Angles()
        self.corner   = corner   if corner   else Angles()
        self.edge     = edge     if edge     else Angles()
        self.face     = face     if face     else Angles()

    def add_item(self, type, site, site_to, ligands, angles) -> None:
        if   type == 'isolated':
            self.isolated.add_item(site, site_to, ligands, angles)
        elif type == 'corner':
            self.corner  .add_item(site, site_to, ligands, angles)
        elif type == 'edge':
            self.edge    .add_item(site, site_to, ligands, angles)
        elif type == 'face':
            self.face    .add_item(site, site_to, ligands, angles)
        else:
            raise ValueError(f'Invalid angle type: {type}')

    def get_site_features(self, site, base=None) -> list:
        result = []
        for feature in self.isolated.get_site_features(site, base=base):
            result.append(
                ['isolated'] + [ [feature[0], ligand, feature[1], angle] for ligand, angle in zip(feature[2], feature[3]) ])
        for feature in self.corner  .get_site_features(site, base=base):
            result.append(
                ['corner'  ] + [ [feature[0], ligand, feature[1], angle] for ligand, angle in zip(feature[2], feature[3]) ])
        for feature in self.edge    .get_site_features(site, base=base):
            result.append(
                ['edge'    ] + [ [feature[0], ligand, feature[1], angle] for ligand, angle in zip(feature[2], feature[3]) ])
        for feature in self.face    .get_site_features(site, base=base):
            result.append(
                ['face'    ] + [ [feature[0], ligand, feature[1], angle] for ligand, angle in zip(feature[2], feature[3]) ])
        return result

## -----------------------------------------------------------------------------

class CryturesSiteIterator():

    def __init__(self, crytures, resolve_elements) -> None:
        self.crytures         = crytures
        self.i                = -1
        self.resolve_elements = resolve_elements
    
    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i >= self.crytures.num_sites:
            raise StopIteration
        else:
            return self.crytures.get_site_features(self.i, self.resolve_elements)

## -----------------------------------------------------------------------------

class Crytures(FancyString, MyMSONable):

    def __init__(self, base = None, distances = None, ces = None, ce_distances = None, ce_angles = None) -> None:
        self.base         = base         if base         else Base()
        self.distances    = distances    if distances    else Distances()
        self.ces          = ces          if ces          else CoordinationEnvironments()
        self.ce_distances = ce_distances if ce_distances else Distances()
        self.ce_angles    = ce_angles    if ce_angles    else CeAngles()

    @property
    def num_sites(self) -> int:
        return len(self.base.oxidations)

    def iterate_over_sites(self, resolve_elements = False):
        return CryturesSiteIterator(self, resolve_elements)

    def get_site_features(self, site, resolve_elements = False) -> str:
        # Check site index first
        if site < 0 or site >= len(self.base.oxidations):
            raise ValueError('Invalid site index given')
        # If resolve_elements is true, pass `base`` object to
        # `get_site_features` method so that it replaces site
        # indices with element names
        if resolve_elements:
            base = self.base
        else:
            base = None
        # Extract features
        features = {}
        features['oxidation'   ] = self.base.oxidations [site]
        features['ion'         ] = self.base.ions       [site]
        features['element'     ] = self.base.elements   [site]
        features['coordinates' ] = self.base.coordinates[site]
        if len(distances := self.distances.get_site_features(site, base=base)) > 0:
            features['distances'   ] = distances
            features['ce'          ] = self.ces         .get_site_features(site)
            features['ce_distances'] = self.ce_distances.get_site_features(site, base=base)
            features['ce_angles'   ] = self.ce_angles   .get_site_features(site, base=base)

        return features
