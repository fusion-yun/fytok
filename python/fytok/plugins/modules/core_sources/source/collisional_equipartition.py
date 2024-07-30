import numpy as np
import scipy.constants
import typing


from spdm.utils.tags import _not_found_
from spdm.core.expression import zero
from fytok.modules.core_profiles import CoreProfiles, CoreProfilesSpecies
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_sources import CoreSources


class CollisionalEquipartition(
    CoreSources.Source,
    identifier="collisional_equipartition",
    code={"name": "collisional_equipartition", "description": "Fusion reaction"},
):
    """Source from Collisional"""

    def execute(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs):
        ii_collision: bool = self.code.parameters.ii_collision
        ie_collision: bool = self.code.parameters.ie_collision
        current = super().execute(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)
        profiles_1d = core_profiles.profiles_1d
        source_1d = current.profiles_1d

        e = scipy.constants.elementary_charge
        ze = -1.0
        me = scipy.constants.electron_mass
        ae = scipy.constants.electron_mass / scipy.constants.atomic_mass

        ne = profiles_1d.electrons.density
        Te = profiles_1d.electrons.temperature
        ve = profiles_1d.electrons.velocity.toroidal

        conductivity_parallel = zero

        source_1d.electrons.energy = zero

        species: typing.List[CoreProfilesSpecies] = [*profiles_1d.ion]

        for i, ion_i in enumerate(species):
            zi = ion_i.z
            ai = ion_i.a
            mi = ion_i.a * scipy.constants.atomic_mass

            ni = ion_i.density
            Ti = ion_i.temperature
            vi = ion_i.velocity.toroidal

            if Ti is _not_found_:
                continue

            # clog_ei = piecewise(
            #     [
            #         (
            #             23 - np.log(zi) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te),
            #             ((Ti * me / mi) < Te) & (Te <= (10 * zi * zi)),
            #         ),
            #         (
            #             24 - 0.5 * np.log(ne * 1.0e-6) + np.log(Te),
            #             (((Ti * me / mi) < (10 * zi * zi)) & ((10 * zi * zi) < Te)),
            #         ),
            #         (
            #             16 - np.log(zi * zi * ae) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te),
            #             Te <= (Ti * me / mi),
            #         ),
            #     ],
            #     name="clog",
            #     label=r"\Lambda_{ei}",
            # )
            delta0 = np.heaviside(Te - Ti * me / mi, 0.5)
            delta1 = np.heaviside(Te - 10 * zi * zi, 0.5)
            delta2 = np.heaviside(Ti * me / mi - 10 * zi * zi, 0.5)

            clog_ei = (
                +(23 - np.log(zi) - 0.5 * np.log(ne * 1.0e-6) + 1.5 * np.log(Te)) * delta0 * (1 - delta1)
                + (24 - 0.5 * np.log(ne * 1.0e-6) + np.log(Te)) * delta1
                + (16 - np.log(zi * zi * ai) - 0.5 * np.log(ni * 1.0e-6) + 1.5 * np.log(Ti)) * (1 - delta0)
            )

            clog_ei._metadata["name"] = "clog"
            clog_ei._metadata["label"] = r"\Lambda_{ei}"

            nv_ei = (
                (3.2e-9 * zi * zi / ai) * (Te ** (-1.5)) * clog_ei * 1.5e-6
            )  # FIXME: 1.5e-6 符合 astra 结果，需要再次复核

            conductivity_parallel += 1.96e-15 * e**2 / me * ne * ne / nv_ei

            nv_ei = ni * ne * nv_ei

            Tie = Ti - Te

            # energy exchange term
            source_1d.ion[ion_i.label].energy -= Tie * nv_ei
            source_1d.electrons.energy += Tie * nv_ei

            # momentum exchange term
            if vi is not _not_found_ and ve is not _not_found_:
                source_1d.ion[ion_i.label].momentum.toroidal -= (vi - ve) * nv_ei
                source_1d.electrons.momentum.toroidal += (ve - vi) * nv_ei

            # collisions frequency and energy exchange terms:
            # @ref NRL 2019 p.34
            for ion_j in species[i + 1 :]:
                if ion_i.label == ion_j.label:
                    continue
                # ion-Ion collisions:

                zj = ion_j.z
                aj = ion_j.a
                mj = ion_j.a * scipy.constants.atomic_mass

                nj = ion_j.density
                Tj = ion_j.temperature
                vj = ion_j.velocity.toroidal

                if Tj is _not_found_:
                    continue

                # Coulomb logarithm:
                clog = (
                    (23 - np.log(zi * zj * (ai + aj)))
                    + np.log(Ti * aj + Tj * ai)
                    - 0.5 * np.log((ni * zi * zi / Ti + nj * zj * zj / Tj) * 1.0e-6)
                )

                nv_ij = (
                    1.8e-19
                    * np.sqrt(mi * mj * 1.0e-6)
                    * (zi * zi * zj * zj)
                    * (((Ti * mj + Tj * mi) * 1.0e-3) ** (-1.5))
                    * clog
                    * 1.0e-6
                )

                # nv_ij = smooth(nv_ij, window_length=3, polyorder=2)

                nv_ij = ni * nj * nv_ij

                Tij = Ti - Tj

                # energy exchange term
                source_1d.ion[ion_i.label].energy -= Tij * nv_ij
                source_1d.ion[ion_j.label].energy += Tij * nv_ij

                # momentum exchange term
                if vi is not _not_found_ and vj is not _not_found_:
                    source_1d.ion[ion_i.label].momentum.toroidal += (vi - vj) * nv_ij
                    source_1d.ion[ion_j.label].momentum.toroidal += (vj - vi) * nv_ij

                # Tij = Expression(deburr, Ti - Tj)

                ##############################
                # 增加阻尼，消除震荡
                # epsilon = 1.0e-10
                # c = (1.5 - 1.0 / (1.0 + np.exp(-np.abs(Ti - Tj) / (Ti + Tj) / epsilon))) * 2
                # Tij = (Ti - Tj) * c
                # if isinstance(c, array_type):
                #     logger.debug((c,))
                ##############################
        # Plasma electrical conductivity:
        source_1d.conductivity_parallel = conductivity_parallel
        return current
