from spdm.util.AttributeTree import AttributeTree, _next_

class EdgeSource(AttributeTree):
    """Edge plasma sources. 
    
        Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)

        .. todo:: 'EdgeSource' IS NOT IMPLEMENTED
    """

    def __init__(self, cache=None, *args, equilibrium=None, rho_tor_norm=None, ** kwargs):
        pass