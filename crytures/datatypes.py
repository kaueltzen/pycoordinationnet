
from typing import NamedTuple

## -----------------------------------------------------------------------------

class Range(NamedTuple):
    start : int
    stop  : int
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

class _Base(NamedTuple):
    sites       : list
    oxidations  : list
    ions        : list
    elements    : list
    coordinates : list

class Base(_Base):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            if 'sites' not in kwargs.keys():
                kwargs['sites'      ] = []
            if 'oxidations' not in kwargs.keys():
                kwargs['oxidations' ] = []
            if 'ions' not in kwargs.keys():
                kwargs['ions'       ] = []
            if 'elements' not in kwargs.keys():
                kwargs['elements'   ] = []
            if 'coordinates' not in kwargs.keys():
                kwargs['coordinates'] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, oxidation, ion, element, coordinates):
        if site != len(self.sites):
            raise ValueError('Invalid order of site features')
        self.sites      .append(site)
        self.oxidations .append(oxidation)
        self.ions       .append(ion)
        self.elements   .append(element)
        self.coordinates.append(coordinates)

## -----------------------------------------------------------------------------

class _Distances(NamedTuple):
    sites       : list
    sites_to    : list
    distances   : list
    indices     : list

class Distances(_Distances, FeatureSequence):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            if 'sites' not in kwargs.keys():
                kwargs['sites'      ] = []
            if 'sites_to' not in kwargs.keys():
                kwargs['sites_to'   ] = []
            if 'distances' not in kwargs.keys():
                kwargs['distances'  ] = []
            if 'indices' not in kwargs.keys():
                kwargs['indices'    ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, site_to, distance):
        super().add_item(site, self.indices)
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

class _CoordinationEnvironments(NamedTuple):
    sites        : list
    ce_symbols   : list
    ce_fractions : list
    csms         : list
    permutations : list
    indices      : list

class CoordinationEnvironments(_CoordinationEnvironments, FeatureSequence):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            if 'sites' not in kwargs.keys():
                kwargs['sites'       ] = []
            if 'ce_symbols' not in kwargs.keys():
                kwargs['ce_symbols'  ] = []
            if 'ce_fractions' not in kwargs.keys():
                kwargs['ce_fractions'] = []
            if 'csms' not in kwargs.keys():
                kwargs['csms'        ] = []
            if 'permutations' not in kwargs.keys():
                kwargs['permutations'] = []
            if 'indices' not in kwargs.keys():
                kwargs['indices'     ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, ce_symbol, ce_fraction, csm, permutation):
        super().add_item(site, self.indices)
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

class _Angles(NamedTuple):
    sites        : list 
    sites_to     : list 
    sites_ligand : list 
    angles       : list 
    indices      : list 

class Angles(_Angles, FeatureSequence):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            if 'sites' not in kwargs.keys():
                kwargs['sites'       ] = []
            if 'sites_to' not in kwargs.keys():
                kwargs['sites_to'    ] = []
            if 'sites_ligand' not in kwargs.keys():
                kwargs['sites_ligand'] = []
            if 'angles' not in kwargs.keys():
                kwargs['angles'      ] = []
            if 'indices' not in kwargs.keys():
                kwargs['indices'     ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, site_to, site_ligand, angle):
        super().add_item(site, self.indices)
        self.sites       .append(site)
        self.sites_to    .append(site_to)
        self.sites_ligand.append(site_ligand)
        self.angles      .append(angle)

    def get_site_features(self, site, base=None):
        result = []
        if site >= len(self.indices):
            return result
        for i in self.indices[site]:
            if base is None:
                r_site        = self.sites       [i]
                r_site_to     = self.sites_to    [i]
                r_site_ligand = self.sites_ligand[i]
            else:
                r_site        = base.elements[self.sites       [i]]
                r_site_to     = base.elements[self.sites_to    [i]]
                r_site_ligand = base.elements[self.sites_ligand[i]]
            result.append([r_site, r_site_ligand, r_site_to, self.angles[i]])
        return result

## -----------------------------------------------------------------------------

class _Crytures(NamedTuple):
    base             : Base
    distances        : Distances
    ces              : CoordinationEnvironments
    ce_distances     : Distances
    ce_angles        : Angles

class Crytures(_Crytures):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            if 'base' not in kwargs.keys():
                kwargs['base'        ] = Base()
            if 'distances' not in kwargs.keys():
                kwargs['distances'   ] = Distances()
            if 'ces' not in kwargs.keys():
                kwargs['ces'         ] = CoordinationEnvironments()
            if 'ce_distances' not in kwargs.keys():
                kwargs['ce_distances'] = Distances()
            if 'ce_angles' not in kwargs.keys():
                kwargs['ce_angles'   ] = Angles()
        return super().__new__(cls, *args, **kwargs)

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
