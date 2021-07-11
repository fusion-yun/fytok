
import collections
import collections.abc
import enum
from math import sqrt

import scipy
from fytok.transport.CoreProfiles import CoreProfiles, CoreProfiles1D
from fytok.transport.CoreSources import CoreSources
from fytok.transport.CoreTransport import CoreTransport, CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.numlib import constants, np
from spdm.numlib.misc import array_like
from spdm.util.logger import logger

from .glf23_mod import glf2d, glf

PI = constants.pi


class GLF23(CoreTransport.Model):
    r"""
        GLF23
        ===============================
            - 2D GLF equations with massless isothermal passing electrons from Waltz et al, Phys. of Plasmas 6(1995)2408
    """

    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": "anomalous", "index": 6,
                            "description": f"anomalous {self.__class__.__name__}"},
             "code": {"name": "glf23"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles,  **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
        core_profiles_1d = core_profiles.profiles_1d
        eq_profiles_1d = equilibrium.time_slice.profiles_1d
        rho_tor = np.asarray(self.profiles_1d.grid_d.rho_tor)
        rho_tor_norm = np.asarray(self.profiles_1d.grid_d.rho_tor_norm)
        psi_norm = np.asarray(self.profiles_1d.grid_d.psi_norm)

        R0 = self.grid.vacuum_toroidal_field.r0
        B0 = self.grid.vacuum_toroidal_field.b0
        rho_bdry = self.grid.rho_tor[-1]

        elongation = eq_profiles_1d.elongation(psi_norm)
        q = eq_profiles_1d.q(psi_norm)
        sqrt_kappa_a = rho_bdry*np.sqrt(elongation)

        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)

        rlte = sqrt_kappa_a * Function(rho_tor, Te).derivative(rho_tor)/Te
        rlne = sqrt_kappa_a * Function(rho_tor, Ne).derivative(rho_tor)/Ne

        Ti = 0  # np.zeros_like(rho_tor)
        Ni = 0  # np.zeros_like(rho_tor)
        Nimp = 0
   
        for ion in core_profiles_1d.ion:
            if not ion.is_impurity:
                Ti = Ti+ion.temperature(rho_tor_norm)*ion.density(rho_tor_norm)
                Ni = Ni+ion.density(rho_tor_norm)
            else:
                Nimp = Nimp+ion.density(rho_tor_norm)
        Ti = Ti/Ni
        # Ti = array_like(rho_tor, Ti)
        # Ni = array_like(rho_tor, Ni)
        rlti = sqrt_kappa_a * Function(rho_tor, Ti).derivative(rho_tor)/Ti
        rlni = sqrt_kappa_a * Function(rho_tor, Ni).derivative(rho_tor)/Ni
        rlnimp = sqrt_kappa_a * Function(rho_tor, Ni).derivative(rho_tor)/Nimp

        dil = 1.0-Ni/Ne
        apwt = Ni/Ne
        aiwt = Nimp/Ne
        taui = Ti/Te
        rmin = rho_tor_norm
        rmaj = R0/rho_bdry

        magnetic_shear = eq_profiles_1d.magnetic_shear(psi_norm)

        zeff = core_profiles_1d.zeff(rho_tor_norm)

        betae = Ne*Te/(B0**2)/(PI*8)

        ##########################################
        # for output
        diff_m = np.zeros_like(rho_tor)
        diff_im_m = np.zeros_like(rho_tor)
        chie_m = np.zeros_like(rho_tor)
        chii_m = np.zeros_like(rho_tor)
        chie_e_m = np.zeros_like(rho_tor)
        exch_m = np.zeros_like(rho_tor)
        etaphi_m = np.zeros_like(rho_tor)
        etapar_m = np.zeros_like(rho_tor)
        etaper_m = np.zeros_like(rho_tor)

        # from nclass import nclass
        iglf = 1
        ######################################################################
        # INPUT
        ########################################################################

        if True:  # set glf parameter
            glf.nmode = 15

            # eigen_gf = 0 use cgg eigenvalue solver (default)
            #           = 1 use generalized tomsqz eigenvalue solver
            #           = 2 use zgeev eigenvalue solver
            glf.eigen_gf = 1
            #  nroot number of equations
            glf.nroot = 8  # 8 for pure plasma, 12 for full impurity dynamics
            #  iflagin(1:20) control flags
            # glf.iflagin_gf = np.zeros(30, dtype=int)
            #   iflagin(1) 0 use ky=ky0; 1 use landau damping point
            glf.iflagin_gf[0] = 1
            #   iflagin(2) 0. local w_d and k_par "2d"; 1 fit to trial function "3d"
            #   iflagin(3) 0,1,and 2 fix up park low high beta and beta lim elong factor
            #   iflagin(4) 0 trapped electron Waltz EoS 1 weiland EoS
            #   iflagin(5) rms_theta 0:fixed; 1 inverse to q/2 ; 2 inverse to root q/2
            #                        3: inverse to xparam(13)*(q/2-1)+1.
            #              5 for retuned rms-theta
            #  xparam(1:20) control parameters
            glf.xparam = np.zeros(20, dtype=float)
            #   xparam(1:2): idelta=xi*xparam(1)+xparam(2) nonadiabatic electron response
            glf.iflagin_gf[1 - 1] = 0
            glf.iflagin_gf[2 - 1] = 1
            #   xparam(3) multiplier park_gf(high betae)/ park_gf(low betae) -1

            glf.iflagin_gf[3 - 1] = 1
            glf.iflagin_gf[4 - 1] = 0
            glf.iflagin_gf[5 - 1] = 3

            glf.xparam_gf[1 - 1] = 0.0
            glf.xparam_gf[2 - 1] = 0
            glf.xparam_gf[3 - 1] = .70
            glf.xparam_gf[4 - 1] = 0.0
            #   xparam(6)+1. is enhancement of xnueff
            glf.xparam_gf[6 - 1] = 0.0
            #   xparam(7) coef of resistivity
            glf.xparam_gf[7 - 1] = 1.0
            #   xparam(8) cut off on rotational stabilization
            glf.xparam_gf[8 - 1] = 0.0
            #   xparam(9)+1. is shape (triangularity) enhancement to beta_crit
            glf.xparam_gf[9 - 1] = 1.0
            #   xparam(10) is high k electron mode enhancement
            glf.xparam_gf[10 - 1] = 0.0
            #   xparam(11:12) lamda parameters
            glf.xparam_gf[11 - 1] = 0.0
            glf.xparam_gf[12 - 1] = 0.0
            #   xparam(13) rms_theta q-dependence
            glf.xparam_gf[13 - 1] = 0.20
            #   xparam(14)  adjustment to gamma_p avoiding negative viscosity
            glf.xparam_gf[14 - 1] = 1.0
            #   xparam(15)   (1+xparam(15)*reps trapped electron fraction
            glf.xparam_gf[15 - 1] = -0.10
            #   xparam(16) rms_theta shat dependence
            glf.xparam_gf[16 - 1] = 0.0
            #   xparam(17) ""
            glf.xparam_gf[17 - 1] = 0.10
            #   xparam(18) rms_theta betae dependence
            glf.xparam_gf[18 - 1] = .00
            #   xparam(19:20)  extra
            glf.xparam_gf[19 - 1] = 0.0
            glf.xparam_gf[20 - 1] = 0.0
            #   xparam(21) 1 add impurity energy diffusivity to ion energy diffusivity
            glf.xparam_gf[21 - 1] = 0.0
            #   xparam(22) >0 keeps gamma_e from changeing spectrum
            glf.xparam_gf[22 - 1] = 0.0
            #   xparam(23) 1. kills kx**2 in k_m**2
            glf.xparam_gf[23 - 1] = 1.0
            #   xparam(24) exb damping model
            glf.xparam_gf[24 - 1] = 0.0
            glf.xparam_gf[25 - 1] = 0.0
            glf.xparam_gf[26 - 1] = 0.0
            glf.xparam_gf[27 - 1] = 0.0
            glf.xparam_gf[28 - 1] = 0.0
            glf.xparam_gf[29 - 1] = 0.0
            glf.xparam_gf[30 - 1] = 0.0

            #  ky0=k_theta*rho_s; k_theta= nq/r; normally 0.3
            glf.xky0_gf = .20

            #  rms_theta width of phi**2 mode function for best fit near pi/3
            glf.rms_theta_gf = scipy.constants.pi/3.0
            glf.park_gf = 0.70
            glf.ghat_gf = 1.0
            glf.gchat_gf = 1.0

            glf.adamp_gf = .500
            glf.cnorm_gf = 100.0

            glf.ikymax_gf = 10
            glf.xkymin_gf = .020
            glf.xkymax_gf = .50

            # .....begin important optional settings
            #
            # turn on self-consistant alpha-stabilization
            #      ialphastab=1
            #
            #      turn on EXB shear stabilization
            #      alpha_e_gf=1. full on ExB shear
            glf.alpha_e_gf = 0.0  # alpha_e
            #     turn on high-k eta-e modes
            glf.xparam_gf[10] = 1.0

        #
        # settings for retuned GLF23 model
        #
        if (iglf == 1):                         # retuned model
            glf.cnorm_gf = 50.0           # ITG normalization (via GYRO runs)
            glf.xparam_gf[10] = 12.0      # ETG normalization (cnorm*xparam(10))
            glf.iflagin_gf[5] = 5         # rms theta fit formula
            glf.xparam_gf[13] = 0.15      # rms_theta q-dependence
            glf.xparam_gf[16] = 0.15      # rms_theta shat dependence
            glf.xparam_gf[17] = 0.25      # rms_theta shat dependence
            glf.xparam_gf[19] = 1.0       # rms_theta alpha dependence
            glf.adamp_gf = .700           # radial mode damping exponent
            glf.alpha_p_gf = 0.350        # parallel velocity shear fit
            glf.park_gf = 0.80            # parallel ion motion fit
            glf.bt_flag = 1               # use real geometry ExB shear

        for idx, x in enumerate(rho_tor):

            #######################################################################
            # rho dependent
            #######################################################################

            #  rlti=a/L_Ti   a/L_f= sqrt(kappa) a d ln f / d rho
            glf.rlti_gf = rlti[idx]
            #  rlte=a/L_Te
            glf.rlte_gf = rlte[idx]
            #  rlne= a/L_ne
            glf.rlne_gf = rlne[idx]
            #  rlni= a/L_ni
            glf.rlni_gf = rlni[idx]
            #  rlnimp= a/L_nim
            glf.rlnimp_gf = rlnimp[idx]
            #  dil=1.-ni_0/ne_0  dilution
            glf.dil_gf = 0.0  # dil[idx]
            #  apwt = ni_0/ne_0
            glf.apwt_gf = 1.0  # apwt[idx]
            #  aiwt = nim_0/ne_0
            glf.aiwt_gf = 0.0  # aiwt[idx]
            #  taui=Ti/Te
            glf.taui_gf = taui[idx]
            #  rmin=r/a
            glf.rmin_gf = rmin[idx]
            #  rmaj=Rmaj/a
            glf.rmaj_gf = rmaj
            # q
            glf.q_gf = q[idx]

            #  xnu=nu_ei/(c_s/a)
            glf.xnu_gf = 0.0  # cxnu*xnu_m[idx]
            #  betae=neTe/(B**2/(8pi))  0 is electrostatic
            glf.betae_gf = betae[idx]

            #  shat= dlnr/drho used only for parallel dynamics part
            glf.shat_gf = magnetic_shear[idx]

            #  alpha local shear parameter or MHD pressure grad (s-alpha diagram)
            #  elong= local elongation or kappa
            glf.elong_gf = elongation[idx]

            #  xwell amount of magnetic well xwell*min(alpha,1)
            #  park=1  (0) is a control parameter to turn on (off) parallel motion
            #       0.405 best at zero beta and 2.5x larger at high beta..see iflagin(3)
            #  ghat=1  (0) is a control parameter to turn on (off) curvature drift
            #  gchat=1 (0) is a control parameter to turn on (off) div EXB motion
            #  adamp= radial mode damping exponent  1/4 < adamp < 3/4
            #       0.25 from direct fit of simulations varying radial mode damping
            #   but 0.75 is better fit to rlti dependence
            #  alpha_star O(1-3)  gyyrobohm breaking coef for diamg. rot. shear
            glf.alpha_star_gf = 0.0
            glf.alpha_mode_gf = 0.0

            #  gamma_star ion diamagnetic rot shear rate in units of c_s/a
            glf.gamma_star_gf = 0.0  # vstarp_m[idx]
            #  alpha_e O(1-3)   doppler rot shear coef
            #  gamma_e    doppler rot shear rate in units of c_s/a
            glf.gamma_e_gf = 0.0  # egamma_m[idx]

            #  alpha_p 1.5  fit for parallel velocity shear effect at rmaj=3 and q=2
            glf.alpha_p_gf = 0.500

            #  gamma_p    parallel velocity shear rate (-d v_phi/ drho) in units of c_s/a
            glf.gamma_p_gf = 0.0  # gamma_p_m[idx]
            #  kdamp model damping normally 0.

            #  atomic number working hydrogen gas
            glf.amassgas_gf = 0.0  # amassgas_exp
            # zimp_exp,       ! effective Z of impurity
            glf.zpmnimp = 1.0
            # amassimp_exp,   ! effective A of impurity
            glf.amassimp_gf = 0.0  # amassimp_exp[idx]

            # impurity dynamics not turned on by default
            # and simple dilution included (idengrad=2, dil_gf=1-nim/nem)
            # to turn on impurity dynamics need to change number of roots
            # supply zimp_exp, amassimp_exp, and fractional density weights
            # apwt_gf and aiwt_gf

            # glf.zimp_gf=6.0
            glf.zimp_gf = zeff[idx]
            # glf.amassimp_gf=12.0

            # zimp_gf=zimp_exp         ! made radially depende FK
            # amassimp_gf=amassimp_exp ! FL made radially dependent

            #######################################################################
            # Model

            #######################################################################
            # Call GLF2D

            glf2d(iglf)

            #
            #######################################################################
            #######################################################################
            # OUTPUT
            #######################################################################
            # yparam(20) output diagnostics
            # logger.debug(glf.yparam)
            # kyf  value of ky used
            # gamma   leading mode growth rate in c_s/a

            # freq    leading mode freq rate in c_s/a
            # ph_m    (e phi /T_e)/(rho_s/a)  saturation value
            # d_hat    plasma diffusivity for ions
            diff_m[idx] = glf.diff_gf
            # d_im_hat    plasma diffusivity for impurities
            diff_im_m[idx] = glf.diff_im_gf

            # chii_hat ion energy diffusivity
            chii_m[idx] = glf.chii_gf
            # chie_hat electron energy diffusivity
            chie_m[idx] = glf.chie_gf
            # exch_hat anomalous e to i energy exchange
            exch_m[idx] = glf.exch_gf
            # eta_par_hat parallel component of toroidal momentum diffusivity
            etapar_m[idx] = glf.eta_par_gf
            # eta_per_hat perpendicular    ""
            etaper_m[idx] = glf.eta_per_gf
            # eta_phi_hat toroidal momentun diffusivity
            etaphi_m[idx] = glf.eta_phi_gf

            chie_e_m[idx] = glf.chie_e_gf

        self.profiles_1d.electrons.particles["d"] = Function(rho_tor_norm, diff_m)
        self.profiles_1d.electrons.energy["d"] = Function(rho_tor_norm, chie_m)

        for ion in core_profiles_1d.ion:
            trans_ion = self.profiles_1d.ion[_next_]
            trans_ion.update({"label": ion.label, "a": ion.a, "z": ion.z})
            if not ion.is_impurity:
                trans_ion.particles["d"] = Function(rho_tor_norm, diff_m)
                trans_ion.energy["d"] = Function(rho_tor_norm,  chii_m)
            else:
                trans_ion.particles["d"] = Function(rho_tor_norm, diff_im_m)


__SP_EXPORT__ = GLF23
