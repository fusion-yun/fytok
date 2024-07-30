from spdm.core.sp_tree import AttributeTree
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
    "dataset_fair"                  : AttributeTree     ,
    "summary"                       : AttributeTree     ,
    "amns_data"                     : AttributeTree     ,
}
# fmt:on
