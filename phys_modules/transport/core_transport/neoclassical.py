import collections

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.util.logger import logger


class NeoClassical(CoreTransport.Model):
    """
        Neoclassical Transport Model
        ===============================
        Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003
        References:
        =============
        - Tokamaks, 3ed,  J.A.Wesson 2003
    """

    def __init__(self, d, *args, **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"neoclassical",
                "index": 5,
                "description": f"{self.__class__.__name__}  Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003"
            }}, d or {}), *args, **kwargs)

    def update(self, *args,
               equilibrium: Equilibrium,
               core_profiles: CoreProfiles,
               **kwargs):

        super().update(*args, core_profiles=core_profiles, **kwargs)

        eV = scipy.constants.electron_volt
        B0 = abs(equilibrium.vacuum_toroidal_field.b0)
        R0 = equilibrium.vacuum_toroidal_field.r0

        core_profile = core_profiles.profiles_1d
        trans = self.profiles_1d

        rho_tor_norm = np.asarray(core_profile.grid.rho_tor_norm)
        rho_tor = np.asarray(core_profile.grid.rho_tor)
        psi_norm = np.asarray(core_profile.grid.psi_norm)
        psi = np.asarray(core_profile.grid.psi)
        q = np.asarray(equilibrium.time_slice.profiles_1d.q(core_profile.grid.psi_norm))

        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        sum1 = 0
        sum2 = 0

        Te = core_profile.electrons.temperature
        Ne = core_profile.electrons.density
        Pe = core_profile.electrons.pressure
        dlnTe = Te.derivative/Te
        dlnNe = Ne.derivative/Ne
        dlnPe = Pe.derivative/Pe
        Te = np.asarray(Te)
        Ne = np.asarray(Ne)
        Pe = np.asarray(Pe)

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        # lnCoul = (14.9 - 0.5*np.log(Ne/1e20) + np.log(Te/1000)) * (Te < 10) +\
        #     (15.2 - 0.5*np.log(Ne/1e20) + np.log(Te/1000))*(Te >= 10)
        # (17.3 - 0.5*np.log(Ne/1e20) + 1.5*np.log(Te/1000))*(Te >= 10)

        # lnCoul = 14
        lnCoul = core_profiles.profiles_1d.coulomb_logarithm
        # electron collision time , eq 14.6.1
        tau_e = np.asarray(1.09e16*((Te/1000)**(3/2))/Ne/lnCoul)

        vTe = np.asarray(np.sqrt(Te*eV/scipy.constants.electron_mass))

        rho_tor[0] = max(1.07e-4*((Te[0]/1000)**(1/2))/B0, rho_tor[0])  # Larmor radius,   eq 14.7.2

        epsilon = np.asarray(rho_tor/R0)
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon**(3/2)
        ###########################################################################################
        #  Sec 14.10 Resistivity
        #
        eta_s = np.asarray(1.65e-9*lnCoul*(Te/1000)**(-3/2))
        nu_e = np.asarray(R0*q/vTe/tau_e/epsilon32)
        Zeff = np.asarray(core_profile.zeff)
        fT = 1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon))
        phi = np.asarray(fT/(1.0+(0.58+0.20*Zeff)*nu_e))
        C = np.asarray(0.56/Zeff*(3.0-Zeff)/(3.0+Zeff))

        eta = eta_s*Zeff/(1-phi)/(1.0-C*phi)*(1.0+0.27*(Zeff-1.0))/(1.0+0.47*(Zeff-1.0))
        trans.conductivity_parallel = 1.0/eta

        ###########################################################################################
        #  Sec 14.12 Bootstrap current
        #
        x = equilibrium.time_slice.profiles_1d.trapped_fraction(psi_norm)  # np.sqrt(2*epsilon)  #
        c1 = (4.0+2.6*x)/(1.0+1.02*np.sqrt(nu_e)+1.07*nu_e)/(1.0 + 1.07 * epsilon32*nu_e)
        c3 = (7.0+6.5*x)/(1.0+0.57*np.sqrt(nu_e)+0.61*nu_e)/(1.0 + 0.61 * epsilon32*nu_e) - c1*5/2

        j_bootstrap = c1*dlnPe + c3 * dlnTe

        ###########################################################################################
        #  Sec 14.11 Chang-0Hinton formula for \Chi_i

        # Shafranov shift
        delta_ = Function(rho_tor, np.array(
            equilibrium.time_slice.profiles_1d.geometric_axis.r(psi_norm)-R0)).derivative

        # impurity ions
        nZI = 0.0

        f1 = (1.0 + (epsilon**2+epsilon*delta_)*3/2 + 3/8*(epsilon**3)*delta_)/(1.0 + epsilon*delta_/2)
        f2 = np.sqrt(1-epsilon**2)*(1 + epsilon*delta_/2)/(1+delta_*(np.sqrt(1-epsilon**2)-1)/epsilon)

        sum1 = 0.0
        sum2 = 0.0
        for idx, sp in enumerate(core_profile.ion):
            Ti = sp.temperature
            Ni = sp.density
            Pi = sp.pressure
            dlnTi = Ti.derivative/Ti
            dlnNi = Ni.derivative/Ni
            dlnPi = Pi.derivative/Pi

            mi = sp.a
            Zi = sp.z_ion_1d
            Zi2 = sp.z_ion_square_1d
            alpha = np.asarray(nZI/(Ni*Zi*Zi))

            # Larmor radius, Tokamaks 3ed, eq 14.7.2
            rho_i = np.asarray(4.57e-3 * np.sqrt(mi*Ti/1000) / B0)

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = np.asarray(6.6e17*np.sqrt(mi)*((Ti/1000)**(3/2))/Ni/(1.1*lnCoul))

            # thermal velocity
            v_Ti = np.sqrt(Ti*(eV/scipy.constants.m_p/mi))

            nu_i = np.asarray(R0*q/epsilon32/v_Ti/tau_i)
            mu_i = np.asarray(nu_i*(1.0+1.54*alpha))

            chi_i = (0.66*(1.0+1.54*alpha)+(1.88*np.sqrt(epsilon)-1.54*epsilon)
                     * (1.0+3.75*epsilon))/(1.0+1.03*np.sqrt(mu_i)+0.31*mu_i)

            chi_i = chi_i * f1 + 0.59*mu_i*epsilon/(1.0+0.74*mu_i*epsilon32)\
                * (1.0 + 1.33*alpha*(1.0+0.60*alpha)/(1.0+1.79*alpha))*(f1-f2)

            chi_i = chi_i/epsilon32*(q**2)*(rho_i**2)/(1.0+0.74*mu_i*epsilon32)

            trans.ion[idx].energy.d = chi_i
            trans.ion[idx].particles.d = chi_i/3.0

            #########################################################################
            #  Sec 14.12 Bootstrap current

            c2 = c1*Ti/Te

            e3n2 = (epsilon ** 3)*(nu_i**2)

            c4 = ((-1.17/(1.0+0.46*x) + 0.35*np.sqrt(nu_i)) / (1 + 0.7*np.sqrt(nu_i)) + 2.1*e3n2) \
                / (1 - e3n2) / (1 + e3n2)*c2

            j_bootstrap = j_bootstrap + c2*dlnPi + c4*dlnTi

            # eq 4.9.2
            # j_bootstrap = j_bootstrap + Ni*Ti*eV*(2.44*dlnNe - 0.42*dlnTi)
            #########################################################################

            sum1 = sum1 + chi_i/3.0*sp.pressure.derivative*Zi/Ti
            sum2 = sum2 + chi_i/3.0*Ni*Zi2 / Ti

        # eq 4.9.2
        # trans.j_bootstrap = (-(q/B0/epsilon12))*j_bootstrap

        trans.j_bootstrap = - j_bootstrap * x/(2.4+5.4*x+2.6*x**2) * Pe * \
            equilibrium.time_slice.profiles_1d.fpol(psi_norm) * q / rho_tor / rho_tor[-1] / (2.0*scipy.constants.pi*B0)

        trans.e_field_radial = sum1/sum2

        ###########################################################################################
        #  Sec 14.12 Bootstrap current
        #　TODO: Sec. 14.12
        ###########################################################################################
        #  Sec 4.6
        #
        return 0.0


__SP_EXPORT__ = NeoClassical