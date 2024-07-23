from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyEntity
from fytok.ontology import edge_profiles


class EdgeProfilesTimeSlice(edge_profiles.EdgeProfilesTimeSlice):
    pass


class EdgeProfiles(IDS, FyEntity, Actor, code={"name": "edge_profiles"}):
    pass
