
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
        if   site == self.n:
            # Given site is still the same
            indices[site] = Range(indices[site][0], indices[site][1]+1)
        elif site == self.n+1:
            self.n += 1
            # Observed a new site
            if len(indices) > 0:
                i_start = indices[self.n-1][1]
                i_stop  = indices[self.n-1][1]+1
            else:
                i_start = 0
                i_stop  = 1
            indices.append(Range(i_start, i_stop))
        else:
            # Seems that we missed a site
            raise ValueError('Invalid order of site features')

## -----------------------------------------------------------------------------

class _Base(NamedTuple):
    sites       : list
    oxidations  : list
    ions        : list
    elements    : list
    coordinates : list

class Base(_Base):
    def __new__(cls, *args, **kwargs):
        kwargs["sites"      ] = []
        kwargs["oxidations" ] = []
        kwargs["ions"       ] = []
        kwargs["elements"   ] = []
        kwargs["coordinates"] = []
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
        kwargs["sites"      ] = []
        kwargs["sites_to"   ] = []
        kwargs["distances"  ] = []
        kwargs["indices"    ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, site_to, distance):
        super().add_item(site, self.indices)
        self.sites    .append(site)
        self.sites_to .append(site_to)
        self.distances.append(distance)
    
    def get_site_features(self, site, base=None):
        if site >= len(self.indices):
            return None
        result = []
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
        kwargs["sites"       ] = []
        kwargs["ce_symbols"  ] = []
        kwargs["ce_fractions"] = []
        kwargs["csms"        ] = []
        kwargs["permutations"] = []
        kwargs["indices"     ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, ce_symbol, ce_fraction, csm, permutation):
        super().add_item(site, self.indices)
        self.sites       .append(site)
        self.ce_symbols  .append(ce_symbol)
        self.ce_fractions.append(ce_fraction)
        self.csms        .append(csm)
        self.permutations.append(permutation)

    def get_site_features(self, site):
        if site >= len(self.indices):
            return None
        result = []
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
        kwargs["sites"       ] = []
        kwargs["sites_to"    ] = []
        kwargs["sites_ligand"] = []
        kwargs["angles"      ] = []
        kwargs["indices"     ] = []
        return super().__new__(cls, *args, **kwargs)

    def add_item(self, site, site_to, site_ligand, angle):
        super().add_item(site, self.indices)
        self.sites       .append(site)
        self.sites_to    .append(site_to)
        self.sites_ligand.append(site_ligand)
        self.angles      .append(angle)

    def get_site_features(self, site, base=None):
        result = []
        for i in self.indices[site]:
            if base is None:
                r_site        = self.sites        [i]
                r_site_to     = self.sites_to     [i]
                r_site_ligand = self.sites_ligands[i]
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
        kwargs["base"        ] = Base()
        kwargs["distances"   ] = Distances()
        kwargs["ces"         ] = CoordinationEnvironments()
        kwargs["ce_distances"] = Distances()
        kwargs["ce_angles"   ] = Angles()
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
        if distances := self.distances.get_site_features(site, base=base) is not None:
            features['distances'   ] = distances
            features['ce'          ] = self.ces         .get_site_features(site)
            features['ce_distances'] = self.ce_distances.get_site_features(site, base=base)
            features['ce_angles'   ] = self.ce_angles   .get_site_features(site, base=base)
        
        return features
