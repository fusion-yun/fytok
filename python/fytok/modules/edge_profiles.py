from spdm.model.actor import Actor
from fytok.modules.utilities import IDS, FyModule
from fytok.ontology import edge_profiles


class EdgeProfilesTimeSlice(edge_profiles.EdgeProfilesTimeSlice):
    pass


class EdgeProfiles(IDS, FyModule, Actor[EdgeProfilesTimeSlice], code={"name": "edge_profiles"}):
    pass
