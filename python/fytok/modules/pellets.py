from __future__ import annotations


from spdm.core.geo_object import GeoObject

from ..ontology import pellets


class Pellets(pellets._T_pellets):
    def __view__(self, view="RZ", **kwargs):
        return {} 
