
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

## -----------------------------------------------------------------------------

class MyMSONable(MSONable):
    def dump(self, filename):
        return dumpfn(self.as_dict(), filename)
    @classmethod
    def load(self, filename):
        return loadfn(filename)

## -----------------------------------------------------------------------------

class FancyString():
    def __str__(self):
        s = f'{self.__class__.__name__}('
        for i, (key, value) in enumerate(self.__dict__.items()):
            if i == 0:
                s += f'{key}={str(value)}'
            else:
                s += f', {key}={str(value)}'
        return s + ')'
    def __repr__(self):
        return str(self)

## -----------------------------------------------------------------------------

class Range(FancyString, MyMSONable):
    start : int
    stop  : int
    def __init__(self, start, stop):
        self.start = start
        self.stop  = stop
    def __iter__(self):
        return iter(range(self.start, self.stop))

## -----------------------------------------------------------------------------

class FeatureSequence():
    n : int

    def __init__(self):
        self.n = -1

    def add_item(self, site, indices):
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
    sites       : list
    oxidations  : list
    ions        : list
    elements    : list
    coordinates : list

    def __init__(self, sites = [], oxidations = [], ions = [], elements = [], coordinates = []):
        self.sites       = sites
        self.oxidations  = oxidations
        self.ions        = ions
        self.elements    = elements
        self.coordinates = coordinates

    def add_item(self, site, oxidation, ion, element, coordinates):
        if site != len(self.sites):
            raise ValueError('Invalid order of site features')
        self.sites      .append(site)
        self.oxidations .append(oxidation)
        self.ions       .append(ion)
        self.elements   .append(element)
        self.coordinates.append(coordinates)

## -----------------------------------------------------------------------------

class Distances(FeatureSequence, FancyString, MyMSONable):
    sites       : list
    sites_to    : list
    distances   : list
    indices     : list

    def __init__(self, sites = [], sites_to = [], distances = [], indices = []):
        self.sites     = sites
        self.sites_to  = sites_to
        self.distances = distances
        self.indices   = indices

    def add_item(self, site, site_to, distance):
        self.sites    .append(site)
        self.sites_to .append(site_to)
        self.distances.append(distance)
    
    def get_site_features(self, site, base=None):
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
    sites        : list
    ce_symbols   : list
    ce_fractions : list
    csms         : list
    permutations : list
    indices      : list

    def __init__(self, sites = [], ce_symbols = [], ce_fractions = [], csms = [], permutations = [], indices = []):
        self.sites        = sites
        self.ce_symbols   = ce_symbols
        self.ce_fractions = ce_fractions
        self.csms         = csms
        self.permutations = permutations
        self.indices      = indices

    def add_item(self, site, ce_symbol, ce_fraction, csm, permutation):
        self.sites       .append(site)
        self.ce_symbols  .append(ce_symbol)
        self.ce_fractions.append(ce_fraction)
        self.csms        .append(csm)
        self.permutations.append(permutation)

    def get_site_features(self, site):
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
    sites     : list 
    sites_to  : list 
    ligands   : list 
    angles    : list 
    indices   : list 

    def __init__(self, sites = [], sites_to = [], ligands = [], angles = [], indices = []):
        self.sites    = sites
        self.sites_to = sites_to
        self.ligands  = ligands
        self.angles   = angles
        self.indices  = indices

    def add_item(self, site, site_to, ligands, angles):
        self.sites    .append(site)
        self.sites_to .append(site_to)
        self.ligands  .append(ligands)
        self.angles   .append(angles)

    def get_site_features(self, site, base=None):
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
    isolated     : Angles
    corner       : Angles
    edge         : Angles
    face         : Angles

    def __init__(self, isolated = Angles(), corner = Angles(), edge = Angles(), face = Angles()) -> None:
        self.isolated = isolated
        self.corner   = corner
        self.edge     = edge
        self.face     = face

    def add_item(self, type, site, site_to, ligands, angles):
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

    def get_site_features(self, site, base=None):
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
    def __init__(self, crytures, resolve_elements):
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
    base             : Base
    distances        : Distances
    ces              : CoordinationEnvironments
    ce_distances     : Distances
    ce_angles        : CeAngles

    def __init__(self, base = Base(), distances = Distances(), ces = CoordinationEnvironments(), ce_distances = Distances(), ce_angles = CeAngles()) -> None:
        self.base         = base
        self.distances    = distances
        self.ces          = ces
        self.ce_distances = ce_distances
        self.ce_angles    = ce_angles

    @property
    def num_sites(self):
        return len(self.base.oxidations)

    def iterate_over_sites(self, resolve_elements=False):
        return CryturesSiteIterator(self, resolve_elements)

    def get_site_features(self, site, resolve_elements=False):
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
