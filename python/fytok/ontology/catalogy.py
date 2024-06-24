from spdm.core.property_tree import PropertyTree
from fytok.modules.utilities import FyComponent, FyActor

# fmt:off
catalogy = {
    "ec_launchers"                  : FyComponent      ,
    "ic_antennas"                   : FyComponent      ,
    "interferometer"                : FyComponent      ,
    "lh_antennas"                   : FyComponent      ,
    "magnetics"                     : FyComponent      ,
    "nbi"                           : FyComponent      ,
    "pellets"                       : FyComponent      ,
    "wall"                          : FyComponent      ,
    "pf_active"                     : FyComponent      ,
    "tf"                            : FyComponent      ,
    "pulse_schedule"                : FyActor          ,
    "equilibrium"                   : FyActor          ,
    "core_profiles"                 : FyActor          ,
    "core_sources"                  : FyActor          ,
    "core_transport"                : FyActor          ,
    "transport_solver_numerics"     : FyActor          ,
    "waves"                         : FyActor          ,
    "dataset_fair"                  : PropertyTree     ,
    "summary"                       : PropertyTree     ,
    "amns_data"                     : PropertyTree     ,
}
# fmt:on
