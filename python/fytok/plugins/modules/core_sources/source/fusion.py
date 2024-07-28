""" Fusion Module

  
"""

import typing
import numpy as np
import scipy.constants
from spdm.core.expression import Variable, zero
from fytok.utils.atoms import nuclear_reaction, Atom
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_profiles import CoreProfiles


PI = scipy.constants.pi


class FusionReaction(
    CoreSources.Source,
    identifier="fusion",
    code={"name": "fusion", "description": "Fusion reaction"},
):
    """[summary]

    Args:
        CoreSources ([type]): [description]


    $\\alpha$输运模型参考[@angioniGyrokineticCalculationsDiffusive2008; @angioniGyrokineticSimulationsImpurity2009]

    * energetic $\\alpha$ particle density $n_{\\alpha}$

    $$
    \\frac{\\partial n_{\\alpha}}{\\partial t}+\\nabla\\left(-D_{\\alpha}\\nabla n_{\\alpha}+Vn_{\\alpha}\\right)=-\\frac{n_{\\alpha}}{\\tau_{sd}^{*}}+n_{D}n_{T}\\left\\langle \\sigma v\\right\\rangle _{DT}
    $$

    * $He$ ash density $n_{He}$

    $$
    \\frac{\\partial n_{He}}{\\partial t}+\\nabla\\left(-D_{He}\\nabla n_{He}+Vn_{He}\\right)=\\frac{n_{\\alpha}}{\\tau_{sd}^{*}}
    $$

    where
    $$
    \\tau_{sd}^{*}=\\ln\\left(v_{\\alpha}^{3}/v_{c}^{3}+1\\right)\\left(m_{e}m_{\\alpha}v_{e}^{3}\\right)/\\left(64\\sqrt{\\pi}e^{4}n_{e}\\ln\\Lambda\\right)
    $$
    is the actual thermalization slowing down time.

    energetic $\\alpha$ particle flux
    $$
    \\frac{R\\Gamma_{\\alpha}}{n_{\\alpha}}=D_{\\alpha}\\left(\\frac{R}{L_{n_{\\alpha}}}C_{p_{\\alpha}}\\right)
    $$
    where
    $$
    D_{\\alpha}=D_{\\text{He}}\\left[0.02+4.5\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)+8\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)^{2}+350\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)^{3}\\right]
    $$
    and
    $$
    C_{p_{\\alpha}}=\\frac{3}{2}\\frac{R}{L_{T_{e}}}\\left[\\frac{1}{\\log\\left[\\left(E_{\\alpha}/E_{c}\\right)^{3/2}+1\\right]\\left[1+\\left(E_{c}/E_{\\alpha}\\right)^{3/2}\\right]}-1\\right]
    $$
    Here $E_{c}$ is the slowing down critical energy. We remind that $E_{c}/E_{\\alpha}=33.05 T_e/E_{\\alpha}$, where $E_{\\alpha}=3500 keV$  is the thirth energy of $\\alpha$ particles.
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    # x = np.linspace(0, 10.0, 256)
    # _x = Variable(0, "x")
    # self._sivukhin = Function(x, 1.0 / (1 + x**1.5)).I(_x) / (_x)
    # self._sivukhin._metadata["name"] = "sivukhin"
    # self._sivukhin._metadata["label"] = "F"

    def execute(self, *args, core_profiles: CoreProfiles, **kwargs):
        current = super().execute(*args, core_profiles=core_profiles, **kwargs)
        profiles_1d = core_profiles.profiles_1d
        heating = self.code.parameters.heating is not False

        rho_tor_norm = profiles_1d.rho_tor_norm

        source_1d = current.profiles_1d
        m_e = scipy.constants.physical_constants["electron mass"][0]
        m_alpha = scipy.constants.physical_constants["alpha particle mass"][0]
        a_alpha = 4
        Te = profiles_1d.electrons.temperature

        fusion_reactions: typing.List[str] = self.code.parameters.fusion_reactions or []

        ne = profiles_1d.electrons.density

        for tag in fusion_reactions:
            if tag != "D(t,n)alpha":
                raise NotImplementedError(f"NOT IMPLEMENTED YET！！ By now only support D(t,n)alpha!")
            reaction = nuclear_reaction[tag]

            r0, r1 = reaction.reactants
            p0, p1 = reaction.products

            pa = Atom[p1].label

            n0 = profiles_1d.ion[r0].density
            n1 = profiles_1d.ion[r1].density

            T0 = profiles_1d.ion[r0].temperature
            T1 = profiles_1d.ion[r1].temperature
            ni = n0 + n1
            Ti = (n0 * T0 + n1 * T1) / ni
            nEP = profiles_1d.ion[p1].density

            E0, E1 = reaction.energy

            lnGamma = 17  # FIXME: 粗略估计

            C = zero
            a_tot = 0

            for label in [r0, r1]:
                ion = profiles_1d.ion[label]
                a_tot += ion.a
                C += ion.density * (ion.z**2) / (ion.a / a_alpha)

            C /= ne

            Ecrit = (Te) * (4 * np.sqrt(m_e / m_alpha) / (3 * np.sqrt(PI) * C)) ** (-2.0 / 3.0)

            # nu_slowing_down = (ni * 1.0e-19 * lnGamma) / (1.99 * ((Ti / 1000) ** (3 / 2)))

            tau_e = (1.99 * ((Te / 1000) ** (3 / 2))) / (ne * 1.0e-19 * lnGamma)

            tau_s = tau_e * np.log((E1 / Ecrit) ** 1.5 + 1) / 3

            S = reaction.reactivities(Ti) * n0 * n1

            if r0 == r1:
                S *= 0.5

            # r_ped = 0.99  #

            # delta = np.heaviside(rho_tor_norm - r_ped, 0.5)

            # S = S * (1 - delta)

            source_1d.ion[r0].particles -= S
            source_1d.ion[r1].particles -= S
            source_1d.ion[p0].particles += S
            source_1d.ion[p1].particles += S - nEP / tau_s
            source_1d.ion[pa].particles += nEP / tau_s

            Efus = S * E1  # nEP * nu_slowing_down

            frac = 0.0

            # if heating:
            # 离子加热分量
            #  [Stix, Plasma Phys. 14 (1972) 367 Eq.15

            frac = self._sivukhin(E1 / Ecrit)

            # 加热离子
            for label in [r0, r1]:
                ion = source_1d.ion[label]

                ion.energy += Efus * frac * ion.a / a_tot

            source_1d.electrons.energy += Efus * (1.0 - frac)

        return current
