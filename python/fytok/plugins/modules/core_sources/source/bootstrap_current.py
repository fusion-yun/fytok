import scipy.constants
import numpy as np

from spdm.core.function import Function

from fytok.utils.logger import logger

from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSources
from fytok.modules.equilibrium import Equilibrium
from fytok.utils.atoms import Atom


class BootstrapCurrent(
    CoreSources.Source,
    identifier="bootstrap",
    code={
        "name": "bootstrap_current",
        "description": "Bootstrap current, based on  Tokamaks, 3ed, sec 14.12 J.A.Wesson 2003",
    },
):
    """Bootstrap current, based on  Tokamaks, 3ed, sec 14.12 J.A.Wesson 2003"""

    def execute(self, *args, **kwargs):
        current = self.execute(*args, **kwargs)

        equilibrium: Equilibrium = self.in_ports.equilibrium
        core_profiles: CoreProfiles = self.in_ports.core_profiles

        source_1d = current.profiles_1d

        eq_1d = equilibrium.profiles_1d

        grid = source_1d.grid

        rho_tor_boundary = grid.rho_tor_boundary

        rho_tor = rho_tor_boundary * x

        psi_norm = Function(grid.rho_tor_norm, grid.psi_norm, label=r"\bar{\psi}")(x)

        eV = scipy.constants.electron_volt

        B0 = equilibrium.vacuum_toroidal_field.b0
        R0 = equilibrium.vacuum_toroidal_field.r0

        q = eq_1d.q(psi_norm)

        # max(np.asarray(1.07e-4*((Te[0]/1000)**(1/2))/B0), rho_tor[0])   # Larmor radius,   eq 14.7.2

        Te = variables.get("electrons/temperature")
        ne = variables.get("electrons/density")
        Pe = scipy.constants.k / scipy.constants.electron_volt * Te * ne
        dlnTe = Te.dln
        dlnne = ne.dln
        dlnPe = dlnne + dlnTe

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        lnCoul = piecewise(
            [
                ((14.9 - 0.5 * np.log(ne / 1e20) + np.log(Te / 1000)), (Te < 10)),
                ((15.2 - 0.5 * np.log(ne / 1e20) + np.log(Te / 1000)), (Te >= 10)),
            ],
            name="clog",
            label=r"\Lambda_{e}",
        )

        # electron collision time , eq 14.6.1
        tau_e = 1.09e16 * ((Te / 1000.0) ** (3 / 2)) / ne / lnCoul

        vTe = np.sqrt(Te * eV / scipy.constants.electron_mass)

        epsilon = rho_tor / R0
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon12**3

        nu_e = R0 * q / vTe / tau_e / epsilon32
        # Zeff = core_profile.zeff

        s = eq_1d.trapped_fraction(psi_norm)  # np.sqrt(2*epsilon)  #
        c1 = (4.0 + 2.6 * s) / (1.0 + 1.02 * np.sqrt(nu_e) + 1.07 * nu_e) / (1.0 + 1.07 * epsilon32 * nu_e)
        c3 = (7.0 + 6.5 * s) / (1.0 + 0.57 * np.sqrt(nu_e) + 0.61 * nu_e) / (
            1.0 + 0.61 * epsilon32 * nu_e
        ) - c1 * 5 / 2

        j_bootstrap = c1 * dlnPe + c3 * dlnTe

        for k, ni in variables.items():
            k_ = k.split("/")
            if not (k_[0] == "ion" and k_[-1] == "density"):
                continue

            s = k[1]
            Ti = variables[f"ion/{s}/temperature"]

            dlnTi = Ti.dln
            dlnNi = ni.dln
            dlnPi = dlnNi + dlnTi
            mi = atoms[s].mass

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = 6.6e17 * np.sqrt(mi) * ((Ti / 1000) ** (3 / 2)) / ni / (1.1 * lnCoul)

            # thermal velocity
            v_Ti = np.sqrt(Ti * (eV / scipy.constants.m_p / mi))

            nu_i = R0 * q / epsilon32 / v_Ti / tau_i

            #########################################################################
            #  Sec 14.12 Bootstrap current

            c2 = c1 * Ti / Te

            e3n2 = (epsilon**3) * (nu_i**2)

            c4 = (
                ((-1.17 / (1.0 + 0.46 * x) + 0.35 * np.sqrt(nu_i)) / (1 + 0.7 * np.sqrt(nu_i)) + 2.1 * e3n2)
                / (1 - e3n2)
                / (1 + e3n2)
                * c2
            )

            j_bootstrap += c2 * dlnPi + c4 * dlnTi
            # eq 4.9.2
            # j_bootstrap = j_bootstrap + Ni*Ti*eV*(2.44*dlnne - 0.42*dlnTi)
            #########################################################################

        # eq 4.9.2
        # src.j_bootstrap = (-(q/B0/epsilon12))*j_bootstrap
        fpol = eq_1d.f(psi_norm)
        j_bootstrap = (
            -j_bootstrap
            * x
            / (2.4 + 5.4 * x + 2.6 * x**2)
            * Pe
            * fpol
            * q
            / x
            / (rho_tor[-1]) ** 2
            / (2.0 * scipy.constants.pi * B0)
        )

        source_1d["j_bootstrap"] = j_bootstrap

        return current
