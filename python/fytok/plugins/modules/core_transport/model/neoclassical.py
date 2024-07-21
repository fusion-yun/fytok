import numpy as np
from scipy import constants
from spdm.numlib.misc import array_like
from spdm.core.function import Function, function_like
from spdm.core.expression import derivative
from spdm.core.entry import _next_
from spdm.core.path import update_tree

from spdm.utils.tags import _not_found_
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransportModel
from fytok.utils.logger import logger


class NeoClassical(
    CoreTransportModel,
    identifier="neoclassical",
    code={
        "name": "neoclassical",
        "description": f" Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003",
    },
):
    """
    Neoclassical Transport Model
    ===============================
    Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003
    References:
    =============
    - Tokamaks, 3ed,  J.A.Wesson 2003
    """

    def execute(self, *args, **kwargs):
        residual = super().execute(*args, **kwargs)

        eV = constants.electron_volt
        B0 = self.in_ports.equilibrium.vacuum_toroidal_field.b0
        R0 = self.in_ports.equilibrium.vacuum_toroidal_field.r0

        core_profiles_1d = self.in_ports.core_profiles.profiles_1d
        equilibrium_1d = self.in_ports.equilibrium.profiles_1d

        rho_tor_norm = self.profiles_1d.grid_d.rho_tor_norm

        rho_tor_lcfs = self.profiles_1d.grid_d.rho_tor_boundary

        psi_norm = self.profiles_1d.grid_d.psi_norm

        q = equilibrium_1d.q(psi_norm)

        epsilon = rho_tor_norm * rho_tor_lcfs / R0
        epsilon12 = np.sqrt(epsilon)
        epsilon32 = epsilon ** (3 / 2)

        # Tavg = np.sum([ion.density*ion.temperature for ion in core_profile.ion]) / \
        #     np.sum([ion.density for ion in core_profile.ion])

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)
        Pe = core_profiles_1d.electrons.pressure(rho_tor_norm)
        dlnTe = derivative(core_profiles_1d.electrons.temperature, rho_tor_norm) / Te / rho_tor_lcfs
        dlnNe = derivative(core_profiles_1d.electrons.density, rho_tor_norm) / Ne / rho_tor_lcfs
        dlnPe = dlnNe + dlnTe

        vTe = np.sqrt(Te * eV / constants.electron_mass)

        # lnCoul = core_profiles_1d.coulomb_logarithm(rho_tor_norm)
        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003
        lnCoul = (14.9 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)) * (Te < 10) + (
            15.2 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)
        ) * (Te >= 10)

        # electron collision time , eq 14.6.1
        tau_e = 1.09e16 * ((Te / 1000) ** (3 / 2)) / Ne / lnCoul

        nu_star_e = R0 * q / vTe / tau_e / epsilon32

        # rho_tor[0] = max(1.07e-4*((Te[0]/1000)**(1/2))/np.abs(B0), rho_tor[0])  # Larmor radius,   eq 14.7.2

        # zeff = core_profiles_1d.zeff(rho_tor_norm)
        zeff = np.zeros_like(rho_tor_norm)
        n_i_total = np.zeros_like(rho_tor_norm)
        ###########################################################################################
        # Sec 14.11 Chang-Hinton formula for \Chi_i
        # Shafranov shift
        delta_ = derivative(equilibrium_1d.geometric_axis.r(psi_norm) - R0, rho_tor_norm) / rho_tor_lcfs

        # impurity ions
        nZ2_imp = np.sum(
            [
                (imp.z_ion_1d(rho_tor_norm) ** 2) * imp.density(rho_tor_norm)
                for imp in core_profiles_1d.ion
                if imp.is_impurity
            ]
        )

        f1 = (1.0 + (epsilon**2 + epsilon * delta_) * 3 / 2 + 3 / 8 * (epsilon**3) * delta_) / (
            1.0 + epsilon * delta_ / 2
        )
        f2 = (
            np.sqrt(1 - epsilon**2)
            * (1 + epsilon * delta_ / 2)
            / (1 + delta_ * (np.sqrt(1 - epsilon**2) - 1) / epsilon)
        )

        ###########################################################################################
        # Sec 14.12 Bootstrap current
        # Tokamak 3ed, 14.10

        # x = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*epsilon12))
        ft_e = 1.0 - (1 - epsilon) ** 2 / np.sqrt(1.0 - epsilon**2) / (1 + 1.46 * np.sqrt(epsilon))

        # fraction of trapped particle
        ft_i = np.sqrt(2 * epsilon)

        c1 = np.array(
            (4.0 + 2.6 * ft_e)
            / (1.0 + 1.02 * np.sqrt(nu_star_e) + 1.07 * nu_star_e)
            / (1.0 + 1.07 * epsilon32 * nu_star_e)
        )
        c3 = np.array(
            (7.0 + 6.5 * ft_e)
            / (1.0 + 0.57 * np.sqrt(nu_star_e) + 0.61 * nu_star_e)
            / (1.0 + 0.61 * epsilon32 * nu_star_e)
            - c1 * 5 / 2
        )

        j_bootstrap = np.asarray(c1 * dlnPe + c3 * dlnTe)

        ###########################################################################################
        # e_field_radial
        sum1 = 0.0
        sum2 = 0.0

        for ion in core_profiles_1d.ion:
            mi = ion.a
            Zi = ion.z_ion_1d(rho_tor_norm)
            Ti = ion.temperature(rho_tor_norm)
            Ni = ion.density(rho_tor_norm)

            # Larmor radius, Tokamaks 3ed, eq 14.7.2
            rho_i = 4.57e-3 * np.sqrt(mi * Ti / 1000) / B0

            # ion collision time Tokamaks 3ed, eq 14.6.2 p730
            tau_i = 6.6e17 * np.sqrt(mi) * ((Ti / 1000) ** (3 / 2)) / Ni / (1.1 * lnCoul)

            # thermal velocity
            v_Ti = np.sqrt(Ti * (eV / constants.m_p / mi))

            nZ2_ion = Ni * Zi * Zi
            zeff += nZ2_ion
            n_i_total += Zi * Ni
            #########################################################################
            # Sec 14.11 Chang-Hinton formula for \Chi_i
            if not ion.is_impurity:
                dlnTi = derivative(ion.temperature, rho_tor_norm) / Ti / rho_tor_lcfs
                dlnNi = derivative(ion.density, rho_tor_norm) / Ni / rho_tor_lcfs
                dlnPi = dlnNi + dlnTi
                alpha = nZ2_imp / nZ2_ion

                nu_star_i = R0 * q / epsilon32 / v_Ti / tau_i

                mu_star_i = nu_star_i * (1.0 + 1.54 * alpha)

                chi_i = (
                    0.66 * (1.0 + 1.54 * alpha) + (1.88 * np.sqrt(epsilon) - 1.54 * epsilon) * (1.0 + 3.75 * epsilon)
                ) / (1.0 + 1.03 * np.sqrt(mu_star_i) + 0.31 * mu_star_i)

                chi_i = chi_i * f1 + 0.59 * mu_star_i * epsilon / (1.0 + 0.74 * mu_star_i * epsilon32) * (
                    1.0 + 1.33 * alpha * (1.0 + 0.60 * alpha) / (1.0 + 1.79 * alpha)
                ) * (f1 - f2)

                chi_i = chi_i / epsilon32 * (q**2) * (rho_i**2) / (1.0 + 0.74 * mu_star_i * epsilon32)

                chi_i = array_like(rho_tor_norm, chi_i)

                self.profiles_1d.ion.put(
                    _next_,
                    {
                        "label": ion.label,
                        "a": ion.a,
                        "z": ion.z,
                        "is_impurity": ion.is_impurity,
                        "energy": {"d": function_like(chi_i, rho_tor_norm)},
                        "particles": {"d": function_like(chi_i / 3.0, rho_tor_norm)},
                    },
                )

                #########################################################################
                # for e_field_radial
                sum1 = sum1 + chi_i / 3.0 * derivative(ion.pressure, rho_tor_norm) / rho_tor_lcfs * Zi / Ti
                sum2 = sum2 + chi_i / 3.0 * nZ2_ion / Ti

            #########################################################################
            # Sec 14.12 Bootstrap current
            e3n2 = (epsilon**3) * (nu_star_i**2)
            c2 = c1 * Ti / Te
            c4 = (
                (
                    (-1.17 / (1.0 + 0.46 * ft_i) + 0.35 * np.sqrt(nu_star_i)) / (1 + 0.7 * np.sqrt(nu_star_i))
                    + 2.1 * e3n2
                )
                / (1 - e3n2)
                / (1 + e3n2)
                * c2
            )

            j_bootstrap += np.asarray(c2 * dlnPi + c4 * dlnTi)

        zeff /= n_i_total

        ###########################################################################################
        #
        self.profiles_1d["e_field_radial"] = sum1 / sum2

        ###########################################################################################
        #  Sec 14.10 Resistivity (spitzer)

        eta_s = 1.65e-9 * lnCoul * (Te / 1000) ** (-3 / 2)

        phi = ft_e / (1.0 + (0.58 + 0.20 * zeff) * nu_star_e)

        C = 0.56 / zeff * (3.0 - zeff) / (3.0 + zeff)

        eta = eta_s * zeff / (1 - phi) / (1.0 - C * phi) * (1.0 + 0.27 * (zeff - 1.0)) / (1.0 + 0.47 * (zeff - 1.0))

        self.profiles_1d["conductivity_parallel"] = function_like(1.0 / eta, rho_tor_norm)

        #########################################################################
        #  Sec 14.12 Bootstrap current

        fpol = equilibrium_1d.fpol(psi_norm)
        j_bootstrap = array_like(
            rho_tor_norm,
            -j_bootstrap
            * ft_e
            / (2.4 + 5.4 * ft_e + 2.6 * ft_e**2)
            * Pe
            * fpol
            * q
            / rho_tor_norm
            / rho_tor_lcfs**2
            / (2.0 * constants.pi * B0),
        )

        self.profiles_1d["j_bootstrap"] = function_like(j_bootstrap, rho_tor_norm)
        self.profiles_1d["j_ohmic"] = function_like(
            core_profiles_1d.e_field.parallel(rho_tor_norm) / eta, rho_tor_norm
        )

        return residual


__SP_EXPORT__ = NeoClassical
