from fytok.utils.base import IDS, FyEntity

from fytok.ontology import distribution_sources


class DistributionSources(FyEntity, IDS, distribution_sources.DistributionSources):
    pass
