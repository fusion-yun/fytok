from functools import cached_property, lru_cache
import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.Function import Function
from spdm.util.logger import logger


class CoreTransport(PhysicalGraph):
    r"""Core plasma transport of particles, energy, momentum and poloidal flux. The transport of particles, energy and momentum is described by
        diffusion coefficients,  :math:`D`, and convection velocities,  :math:`v`. These are defined by the total fluxes of particles, energy and momentum, across a
        flux surface given by : :math:`V^{\prime}\left[-DY^{\prime}\left|\nabla\rho_{tor,norm}\right|^{2}+vY\left|\nabla\rho_{tor,norm}\right|\right]`,
        where  :math:`Y` represents the particles, energy and momentum density, respectively, while  :math:`V` is the volume inside a flux surface, the primes denote derivatives with respect to :math:`\rho_{tor,norm}` and
        :math:`\left\langle X\right\rangle` is the flux surface average of a quantity  :math:`X`. This formulation remains valid when changing simultaneously  :math:`\rho_{tor,norm}` into :math:`\rho_{tor}`
        in the gradient terms and in the derivatives denoted by the prime. The average flux stored in the IDS as sibling of  :math:`D` and  :math:`v` is the total
        flux described above divided by the flux surface area :math:`V^{\prime}\left\langle \left|\nabla\rho_{tor,norm}\right|\right\rangle` . Note that the energy flux includes the energy transported by the particle flux.

        Attributes :
            profiles_1d
    """
    IDS = "core_transport"

    def __init__(self,  *args,   **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, time=None, ** kwargs):
        logger.debug(f"Update {self.__class__.__name__} [time={time}] at: Do nothing")

    @property
    def grid(self):
        return self._parent.radial_grid

    class TransportCoeff(PhysicalGraph):
        def __init__(self,   *args, **kwargs):
            super().__init__(*args,   **kwargs)
            self.d = Function(self._parent.grid_d.rho_tor_norm, self["d"])
            self.v = Function(self._parent.grid_v.rho_tor_norm, self["v"])
            self.flux = Function(self._parent.grid_flux.rho_tor_norm, self["flux"])

    class Particle(PhysicalGraph):
        def __init__(self, *args,  **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def particles(self):
            return CoreTransport.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransport.TransportCoeff(self["energy"], parent=self._parent)

    class Ion(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    class Electrons(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    class Neutral(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    @property
    def grid_d(self):
        """Grid for effective diffusivities and parallel conductivity"""
        return self._parent.radial_grid

    @property
    def grid_v(self):
        """ Grid for effective convections
            Todo :
                FIXME
        """
        return self._parent.radial_grid

    @property
    def grid_flux(self):
        """ Grid for fluxes
            Todo :
                FIXME
        """
        return self._parent.radial_grid

    @cached_property
    def ion(self):
        """ Ion　: Transport coefficients related to the various ion species """
        return CoreTransport.Ion(self['ion'], parent=self)

    @cached_property
    def electrons(self):
        """ Electrons :　Transport quantities related to the electrons"""
        return CoreTransport.Electrons(self['electrons'], parent=self)

    @cached_property
    def neutral(self):
        """ Neutral : Transport coefficients related to the various neutral species"""
        return CoreTransport.Neutral(self['neutral'], parent=self)

    @cached_property
    def total_ion_energy(self):
        """
            CoreTransport.TransportCoeff : Transport coefficients for the total 
             (summed over ion species) energy equation
        """
        return CoreTransport.TransportCoeff(self["total_ion_energy"], parent=self)

    @cached_property
    def momentum_tor(self):
        """
            CoreTransport.TransportCoeff : Transport coefficients for total toroidal momentum equation
        """
        return CoreTransport.TransportCoeff(self["momentum_tor"], parent=self)

    @cached_property
    def conductivity_parallel(self):
        return Function(self.grid_d.rho_tor_norm, self["conductivity_parallel"] or 0.0)

    @cached_property
    def e_field_radial(self):
        """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
        return Function(self.grid_d.rho_tor_norm, self["e_field_radial"])
