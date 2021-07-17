import collections
import collections.abc

from fytok.transport.CoreProfiles import CoreProfiles, CoreProfiles1D
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.numlib import constants, np
from spdm.numlib.misc import array_like
from spdm.util.logger import logger

from .mod_nclass import nclass_mod

NCLASS_MSG = [
    "iflag=-4 warning: no viscosity",
    "iflag=-3 warning: no banana viscosity",
    "iflag=-2 warning: no Pfirsch-Schluter viscosity",
    "iflag=-1 warning: no potato orbit viscosity",
    "iflag=0 no warnings or errors               ",
    "iflag=1 error: order of v moments to be solved must be 2 or 3",
    "iflag=2 error: number of species must be 1< m_i < mx_mi+1",
    "iflag=3 error: number of species must be 0 < m_z < mx_mz+1",
    "iflag=4 error: number of species must be 1 < m_s < mx_ms+1",
    "iflag=5 error: inversion of flow matrix failed",
    "iflag=6 error: trapped fraction must be 0.0.le.p_ft.le.1.0",
]


class NClass(CoreTransportModel):
    def __init__(self, d: collections.abc.Mapping = None,  *args, **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": "neoclassical", "index": 5,  "description": f"{self.__class__.__name__}"},
             "code": {"name": "nclass"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles,  **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
        eq_profile = equilibrium.time_slice.profiles_1d
        core_profile = core_profiles.profiles_1d
        core_trans_prof = self.profiles_1d

        # core_trans_prof.ion = [{
        #     "label": ion.label,
        #     "z_ion": ion.z_ion,
        #     "neutral_index": ion.neutral_index,
        #     "element": ion.element,
        #     "multiple_states_flag": ion.multiple_states_flag,
        #     "state": ion.state
        # } for ion in core_profile.ion]

        # ----------------------------------------------------------------------
        # Model options
        #  kboot           : internal option for metrics
        #                     = 1 -> Hirshman, Sigmar model: fm[k]=Lck^*,           xngrth=<(n.gr(B))^2>/<B^2>.
        #                     = 2 -> Kessel model:           fm[1]=Rq/eps**3/2.     xngrth not used.
        #                     =   -> Shaing, et al model: fm[k]=Fk,                 xngrth=<n.gr(theta)>
        # ----------------------------------------------------------------------
        # kboot   = core_transport.codeparam.parameters[1]
        kboot = 0
        #    by default :     kboot = 0nMaxSpec

        # -------------------------------------------------------
        #  k_order        : order of v moments to be solved (-)
        #                   = 2 -> u and q
        #                   = 3 -> u, q, and u2
        #                   =   -> else error
        # -------------------------------------------------------
        k_order = 3

        # ----------------------------------------------------
        #  l_potato       :  option to include potato orbits (-)
        #                   = 0 -> off
        #                   =   -> else on
        l_potato = 1
        # ------------------------------------------------
        #  l_banana       : option to include banana viscosity (-)
        #                 = 0 -> off
        #                 =   -> else on
        l_banana = 1
        # ------------------------------------------------------------------
        #  l_pfirsch      : option to include Pfirsch-Schluter viscosity (-)
        #                 = 0 -> off
        #                 =   -> else on
        l_pfirsch = 1

        m_i = 1+len(core_profile.ion)

        m_z = np.sum([ion.z_ion for ion in core_profile.ion])
        # ------------------------------------------------------------------------
        #  c_den          : density cutoff below which species is ignored (/m**3)
        # ------------------------------------------------------------------------
        c_den = 1.0e10
        # -----------------------------------------------------------------------------
        #  p_grphi        : radial electric field Phi' (V/rho)
        #  p_gr2phi       : radial electric field gradient Psi'(Phi'/Psi')' (V/rho**2)
        # -----------------------------------------------------------------------------

        psi_norm = eq_profile.psi_norm
        rho_tor_norm = core_profile.grid.rho_tor_norm
        rho_tor = core_profile.grid.rho_tor
        rho_tor_bdry = rho_tor[-1]

        bt0_pr = eq_profile.fpol * eq_profile.gm1 / eq_profile.gm9
        gph_pr = eq_profile.gm1*eq_profile.dvolume_drho_tor * rho_tor_bdry

        b0 = equilibrium.vacuum_toroidal_field.b0
        r0 = equilibrium.vacuum_toroidal_field.r0

        r_inboard = eq_profile.r_inboard
        r_outboard = eq_profile.r_outboard

        xr0_pr = (r_inboard+r_outboard)*0.5

        #  external poloidal current has a flux surface (A)|
        xfs_pr = xr0_pr*bt0_pr * (2.0*constants.pi/constants.mu_0)

        # Electron,Ion densities, temperatures and mass

        Ts = [core_profile.electrons.temperature, *[ion.temperature for ion in core_profile.ion]]
        dTs = [t.derivative() for t in Ts]

        ns = [core_profile.electrons.density, *[ion.density for ion in core_profile.ion]]

        dpsi_drho_tor_norm = eq_profile.dpsi_drho_tor_norm

        dPs = [core_profile.electrons.pressure.derivative() * dpsi_drho_tor_norm,
               *[ion.pressure.derivative() * dpsi_drho_tor_norm for ion in core_profile.ion]]

        amu = [constants.m_e/constants.m_u,   # Electron mass in amu
               * [sum([(element.a*element.atoms_n) for element in ion.element]) for ion in core_profile.ion]]

        q = eq_profile.q

        dq_drho_tor = q.derivative()*eq_profile.dpsi_drho_tor
        dphi_drho_tor = q*eq_profile.dpsi_drho_tor
        fpol = eq_profile.fpol
        kappa = eq_profile.elongation

        c_potb = kappa[0]*b0/2.0/q[0]
        c_potl = r0*q[0]

        psi_norm = Function(core_trans_prof.grid_d.rho_tor_norm, core_trans_prof.grid_d.psi_norm)

        flag = True

        ele = core_trans_prof.electrons

        x_rho = core_trans_prof.grid_d.rho_tor_norm

        profiles = {
            "electrons": {
                "particles": {
                    "flux": array_like(x_rho, 0),
                    "d": array_like(x_rho, 0),
                    "v": array_like(x_rho, 0),
                },
                "energy": {
                    "flux": array_like(x_rho, 0),
                    "d": array_like(x_rho, 0),
                    "v": array_like(x_rho, 0),
                },
            },
            "ion": []
        }
        for ion in core_trans_prof.ion:
            profiles["ion"].append({
                "label": ion.label,
                "particles": {
                    "flux": array_like(x_rho, 0),
                    "d": array_like(x_rho, 0),
                    "v": array_like(x_rho, 0),
                },
                "energy": {
                    "flux": array_like(x_rho, 0),
                    "d": array_like(x_rho, 0),
                    "v": array_like(x_rho, 0),
                },
            })

        # Set input for NCLASS
        for ipr, x in enumerate(x_rho):
            x_psi = psi_norm(x)
            xqs = q(x_psi)
            # Geometry and electrical field calculations

            dpsi_drho_tor = eq_profile.dpsi_drho_tor(x_psi)
            jparallel = eq_profile.j_parallel(x_psi)*b0
            dphi_dpsi = eq_profile.dphi_dpsi(x_psi)
            grad_rho_tor2 = eq_profile.gm3(x_psi)

            #-------------#
            #Call NCLASS  #
            #-------------#
            qsf = q(x_psi)
            fex_iz = np.zeros([3, 3])
            fex_iz[0, 1] = 0.0
            fex_iz[1, 1] = 0.0
            fex_iz[2, 1] = 0.0
            dencut = 1e10
            xr0 = xr0_pr(x_psi)
            rminx = x*rho_tor_bdry

            xeps = rminx/xr0
            xgph = gph_pr(x_psi)/4/(constants.pi**2)
            xfs = xfs_pr(x_psi)

            if (xgph == 0):
                xgph = 0.01

            p_fm = [0.0,  0.0, 0.0]

            if (kboot == 1):
                # -----------------
                #         Hirshman
                # -----------------
                p_ngrth = (xeps/(xqs*xr0))**2/2.0
                p_fm[0] = xqs*xr0
            elif (kboot == 2):
                # ---------------
                #         Kessel
                # ---------------
                p_ngrth = 0.0
                p_fm[0] = xqs*xr0/xeps**1.5
            else:
                # ---------------
                #         Shaing
                # ---------------
                p_ngrth = 1.0/(xqs*xr0)
                if (xeps > 0.0):
                    eps2 = xeps**2
                    b = np.sqrt(1.0 - eps2)
                    a = (1.0 - b)/xeps
                    c = (b**3.0)*(xqs*xr0)**2

                    p_fm[0] = 1*a**(2*1)*(1.0 + 1*b)/c
                    p_fm[2] = 2*a**(2*2)*(1.0 + 2*b)/c
                    p_fm[1] = 3*a**(2*3)*(1.0 + 3*b)/c

            inputs = (
                m_i,                                              # number of isotopes (> 1) [-]
                m_z,                                              # highest charge state [-]
                eq_profile.gm5(x_psi),                            # <B**2> [T**2]
                eq_profile.gm4(x_psi),                            # <1/B**2> [1/T**2]
                core_profile.e_field.parallel(x_psi)*b0,          # <E.B> [V*T/m]
                constants.mu_0*fpol(x_psi) / dpsi_drho_tor,       # mu_0*F/(dPsi/dr) [rho/m]
                p_fm,                                             # poloidal moments of drift factor for PS [/m**2]
                eq_profile.trapped_fraction(x_psi),               # trapped fraction [-]
                eq_profile.gm6(x_psi),                            # <grad(rho)**2/B**2> [rho**2/m**2/T**2]
                dphi_drho_tor(x_psi),                             # potential gradient Phi' [V/rho]
                ((dpsi_drho_tor**2) * dq_drho_tor(x_psi)),        # Psi'(Phi'/Psi')' [V/rho**2]
                p_ngrth,                                          # <n.grad(Theta)> [/m]
                amu,                                              # atomic mass number [-]
                [dT(x_psi) for dT in dTs],                        # temperature gradient [keV/rho]
                [T(x_psi) for T in Ts],                           # temperature [keV]
                [n(x_psi) for n in ns],                           # density [/m**3]
                fex_iz[:, :],                                     # moments of external parallel force [T*n/m**3]
                [dp(x_psi) for dp in dPs],                        # pressure gradient [keV/m**3/rho]
                ipr,                                              #
                l_banana,                                         # option to include banana viscosity [logical]
                # option to include Pfirsch-Schluter viscosity [logical]
                l_pfirsch,
                l_potato,                                         # option to include potato orbits [logical]
                k_order,                                          # order of v moments to be solved [-]
                                                                  #            =2 u and q (default)
                                                                  #            =3 u, q, and u2
                                                                  #            =else error
                # density cutoff below which species is ignored (default 1.e10) [/m**3]
                c_den,
                c_potb,                                           # kappa(0)*Bt(0)/[2*q(0)**2] [T]
                c_potl,                                           # q(0)*R(0) [m]
            )

            outputs = nclass_mod.nclass(*inputs)

            # logger.debug(outputs)

            (
                iflag,      # " int"
                p_etap,     # - parallel electrical resistivity [Ohm*m]
                p_jbbs,     # - <J_bs.B> [A*T/m**2]
                p_jbex,     # - <J_ex.B> current response to fex_iz [A*T/m**2]
                p_jboh,     # - <J_OH.B> Ohmic current [A*T/m**2]
                m_s,        # - number of species [ms>1]
                jm_s,       # - isotope number of s [-]
                jz_s,       # - charge state of s [-]
                bsjbp_s,    # - <J_bs.B> driven by unit p'/p of s [A*T*rho/m**2]
                bsjbt_s,    # - <J_bs.B> driven by unit T'/T of s [A*T*rho/m**2]
                glf_s,      # - radial particle flux comps of s [rho/m**3/s]
                            #   m=1, banana-plateau, p' and T'
                            #   m=2, Pfirsch-Schluter
                            #   m=3, classical
                            #   m=4, banana-plateau, <E.B>
                            #   m=5, banana-plateau, external parallel force fex_iz
                dn_s,       # -diffusion coefficients (diag comp) [rho**2/s]
                vnnt_s,     # -convection velocity (off diag p',T' comps) [rho/s]
                vneb_s,     # -<E.B> particle convection velocity [rho/s]
                vnex_s,     # -external force particle convection velocity [rho/s]
                dp_ss,      # -diffusion coefficient of s2 on p'/p of s1 [rho**2/s]
                dt_ss,      # -diffusion coefficient of s2 on T'/T of s1 [rho**2/s]
                            # ---------------------------------------------------------
                            #          Momentum equation
                upar_s,     # UPAR_S(3,m,s)-parallel flow of s from force m [T*m/s]
                            #                m=1, p', T', Phi'
                            #                m=2, <E.B>
                            #                m=3, fex_iz
                utheta_s,   # UTHETA_S(3,m,s)-poloidal flow of s from force m [m/s/T]
                            #                  m=1, p', T'
                            #                  m=2, <E.B>
                            #                  m=3, fex_iz
                            # ---------------------------------------------------------
                            #           Energy equation
                qfl_s,      # -radial heat conduction flux comps of s [W*rho/m**3]
                            #             m=1, banana-plateau, p' and T'
                            #             m=2, Pfirsch-Schluter
                            #             m=3, classical
                            #             m=4, banana-plateau, <E.B>
                            #             m=5, banana-plateau, external parallel force fex_iz
                chi_s,      # -conduction coefficients (diag comp) [rho**2/s]
                vqnt_s,     # -conduction velocity (off diag p',T' comps) [rho/s]
                vqeb_s,     # -<E.B> heat convection velocity [rho/s]
                vqex_s,     # -external force heat convection velocity [rho/s]
                chip_ss,    # -heat cond coefficient of s2 on p'/p of s1 [rho**2/s]
                chit_ss,    # -heat cond coefficient of s2 on T'/T of s1 [rho**2/s]
                            # ---------------------------------------------------------
                            #           Friction coefficients
                calm_i,     # test particle (tp) friction matrix [-]
                caln_ii,    # field particle (fp) friction matrix [-]
                capm_ii,    # tp eff friction matrix [kg/m**3/s]
                capn_ii,    # fp eff friction matrix [kg/m**3/s]
                            # ---------------------------------------------------------
                            #           Viscosity coefficients
                ymu_s,      # normalized viscosity for s [kg/m**3/s]
                            # ---------------------------------------------------------
                            #           Miscellaneous
                sqz_s,      # orbit squeezing factor for s [-]
                xi_s,       # charge weighted density factor of s [-]
                tau_ss,     # 90 degree scattering time [s]
            ) = outputs
            
            if iflag != 0:
                msg = NCLASS_MSG[iflag+4]
                if iflag < 0:
                    logger.warning(f"NCLASS (i={ipr}) {msg} ")
                else:
                    logger.error(f"NCLASS (i={ipr}) {msg} ")
                if flag is True:
                    flag = []
                flag.append((ipr, msg))

            # Update utheta with edotb rescaling
            utheta_s[:, 2, :] = p_etap*(jparallel - p_jbbs)*utheta_s[:, 2, :]

            # # Electron particle flux
            profiles["electrons"]["particles"]["flux"][ipr] = np.sum(glf_s[:, 0])
            profiles["electrons"]["particles"]["d"][ipr] = dn_s[0]/grad_rho_tor2
            profiles["electrons"]["particles"]["v"][ipr] = vnnt_s[0] + vneb_s[0]*p_etap*(jparallel - p_jbbs)
            # Electrons  heat flux
            profiles["electrons"]["energy"]["flux"][ipr] = np.sum(qfl_s[:, 0])
            profiles["electrons"]["energy"]["d"][ipr] = chi_s[0]/grad_rho_tor2
            profiles["electrons"]["energy"]["v"][ipr] = vqnt_s[0] + vqeb_s[0]*p_etap*(jparallel - p_jbbs)

            # core_trans_prof.chieff[ipr] = chi_s[1]/grad_rho_tor2  # need to set 0.0

            # Ion heatfluxes
            for k, sp in enumerate(core_trans_prof.ion):
                # ion particle fluxes
                profiles["ion"][k]["particles"]["flux"][ipr] = np.sum(glf_s[:, k + 1])
                profiles["ion"][k]["particles"]["d"][ipr] = dn_s[k + 1]/grad_rho_tor2
                profiles["ion"][k]["particles"]["v"][ipr] = vnnt_s[k + 1] + vneb_s[k + 1]*p_etap*(jparallel - p_jbbs)
                # Ion heat flux
                profiles["ion"][k]["energy"]["flux"][ipr] = np.sum(qfl_s[:, k + 1])
                profiles["ion"][k]["energy"]["d"][ipr] = chi_s[k + 1]/grad_rho_tor2
                profiles["ion"][k]["energy"]["v"][ipr] = vqnt_s[k + 1] + vqeb_s[k + 1]*p_etap*(jparallel - p_jbbs)
                # sp.deff[ipr] = dn_s[k + 1]/grad_rho_tor2

            # Ionic rotational  momentum transport
            # sp.momentum.d[ipr] = 0.0  # Need to set

            # update poloidal velocities
            # rotation_frequency_tor_sonic
            # profiles_rm.vpol[ipr] = np.sum(utheta_s[1, 1:3, k + 1])*fpol[i] / r0/fhat

            # Update toroidal velocities
            # if k != 1:
            # profiles_rm.vtor[ipr] = r0*p_fpolhat[i]/p_fpol[i]*profiles_rm.e_rad[i] +\
            #     p_fpolhat[i]*profiles_rm.vpol[ipr] - r0 * \
            #     p_fpolhat[i]/p_fpol[i]*e_r[3]

            # core_trans_prof.electrons.particles.d.eff[ipr] = dn_s[1]/grad_rho_tor2  # need to set

            # rotational momentum transport
            # core_trans_prof.momentum_tor.d[ipr] = 0.0

            # resistivity and <j dot B>
            # core_trans_prof.conductivity_parallel[ipr] = 1.0 / p_etap
            # core_trans_prof.j_bootstrap[ipr] = p_jbbs/b0

            # Recalculate E_r for storage
            # core_profile.e_field.radial[ipr] = NotImplemented  # p_fpol[i]/r0/fhat*profiles_rm.vtor[i, 1]
            # core_profile.e_field.poloidal[ipr] = NotImplemented  # -p_fpol[i]/r0*profiles_rm.vpol[i, 1]
            # core_profile.e_field.toroidal[ipr] = NotImplemented  # p_fpol[i]/r0/fhat*profiles_rm.vtor[i, 1]

        core_trans_prof.electrons.particles["d"] = Function(x_rho, profiles["electrons"]["particles"]["d"])
        core_trans_prof.electrons.particles["v"] = Function(x_rho, profiles["electrons"]["particles"]["v"])
        core_trans_prof.electrons.particles["flux"] = Function(x_rho, profiles["electrons"]["particles"]["flux"])
        core_trans_prof.electrons.energy["d"] = Function(x_rho, profiles["electrons"]["energy"]["d"])
        core_trans_prof.electrons.energy["v"] = Function(x_rho, profiles["electrons"]["energy"]["v"])
        core_trans_prof.electrons.energy["flux"] = Function(x_rho, profiles["electrons"]["energy"]["flux"])

        for idx, ion in enumerate(core_trans_prof.ion):
            ion.particles["d"] = Function(x_rho, profiles["ion"][idx]["particles"]["d"])
            ion.particles["v"] = Function(x_rho, profiles["ion"][idx]["particles"]["v"])
            ion.particles["flux"] = Function(x_rho, profiles["ion"][idx]["particles"]["flux"])
            ion.energy["d"] = Function(x_rho, profiles["ion"][idx]["energy"]["d"])
            ion.energy["v"] = Function(x_rho, profiles["ion"][idx]["energy"]["v"])
            ion.energy["flux"] = Function(x_rho, profiles["ion"][idx]["energy"]["flux"])

        # Extend to edge values
        # core_trans_prof.chi[-1, :] = core_trans_prof.chi[-1 - 1, :]
        # core_trans_prof.Vt[-1, :] = core_trans_prof.Vt[-1 - 1, :]
        # core_trans_prof.heat_fluxes[-1, :] = core_trans_prof.heat_fluxes[-1 - 1, :]
        # core_trans_prof.chieff[-1, :] = core_trans_prof.chieff[-1 - 1, :]

        # core_transport.d[-1, :] = core_transport.d[-1 - 1, :]
        # core_transport.Vp[-1, :] = core_transport.Vp[-1 - 1, :]
        # core_transport.particle_fluxes[-1, :] = core_transport.particle_fluxes[-1 - 1, :]
        # core_transport.deff[-1, :] = core_transport.deff[-1 - 1, :]

        # # rotational  momentum core_transport
        # core_transport.chimom[-1, :] = core_transport.chimom[-1 - 1, :]

        # core_transport.resistivity[-1] = core_transport.resistivity[-1 - 1]
        # core_transport.jboot[-1] = core_transport.jboot[-1 - 1]

        # Copy local values to profiles
        # core_profile.vloop[ipr] = profiles_rm.vpol[1, :]
        return 0.0
