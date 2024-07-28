import collections.abc
import typing
import numpy as np
import scipy.constants
from spdm.core.htree import Dict, List
from spdm.core.expression import Expression
from spdm.core.sp_tree import annotation, sp_property, sp_tree, SpTree
from spdm.core.sp_tree import AttributeTree
from spdm.core.path import update_tree

# from spdm.utils.type_hint import get_args
from spdm.utils.tags import _not_found_

#################################################
# TODO: 需要 AMNS 数据库接口
#################################################
_predef_atoms = {
    "electron": {
        "label": "e",
        "z": -1,
        "a": scipy.constants.electron_mass / scipy.constants.atomic_mass,
        "mass": scipy.constants.electron_mass,
    },
    "e": "electron",
    "electrons": "electron",
    "n": {
        "label": "n",
        "z": 0,
        "a": 1,
        "mass": scipy.constants.physical_constants["neutron mass"][0],
    },
    "p": {
        "label": "p",
        "z": 1,
        "a": 1,
        "mass": scipy.constants.physical_constants["proton mass"][0],
    },
    "H": {
        "label": "H",
        "z": 1,
        "a": 1,
        "mass": scipy.constants.physical_constants["proton mass"][0],
    },
    "D": {
        "label": "D",
        "z": 1,
        "a": 2,
        "mass": scipy.constants.physical_constants["deuteron mass"][0],
    },
    "T": {
        "label": "T",
        "z": 1,
        "a": 3,
        "mass": scipy.constants.physical_constants["triton mass"][0],
    },
    "3He": {
        "label": "3He",
        "z": 2,
        "a": 3,
        "mass": scipy.constants.physical_constants["helion mass"][0],
    },
    "He": {
        "label": "He",
        "z": 2,
        "a": 4,
        "mass": scipy.constants.physical_constants["alpha particle mass"][0],
    },
    "alpha": "He",
    "Be": {"label": "Be", "z": 4, "a": 9},
    "Ar": {"label": "Ar", "z": 18, "a": 40},
}


class Atom(SpTree):
    """Atom database"""

    def __init__(self, data: str | dict, **kwargs):
        if data is None or data is _not_found_:
            data = kwargs.pop("label", None)
        data_ = data
        while isinstance(data, str):
            data = _predef_atoms.get(data, _not_found_)
            if data == data_:
                raise RuntimeError(f"Atom {data_} not found in the database")

        if isinstance(data, Atom):
            data = data._cache

        super().__init__(data, **kwargs)

    label: str
    z: float
    a: float
    mass: float
    elements: List[AttributeTree]


# class Atoms(Dict[Atom]):
#     """Atoms database"""

#     def __get_node__(self, key: str, default_value=_not_found_) -> Atom:
#         if not isinstance(key, str):
#             raise RuntimeError(f"Atom key must be a string, not {key} {self._cache}")

#         if key.startswith("ion/"):
#             key = key[4:]

#         return super().__get_node__(key)


# atoms = Atoms(_predef_atoms)


def get_species(species):
    if isinstance(species, str):
        return atoms.get(species, {"label": species})

    elif isinstance(species, collections.abc.Sequence):
        return [atoms.get(s, {"label": s}) for s in species]

    elif isinstance(species, collections.abc.Mapping):
        label = species.get("label", None)
        if label is None:
            raise ValueError(f"Species {species} must have a label")
        else:
            return update_tree(species, atoms.get(label, {"label": label}))
    else:
        raise TypeError(f"Unknown species type: {type(species)}")


@sp_tree
class Reaction:
    reactants: tuple
    products: tuple
    reactivities: Expression = annotation(label=r"\sigma")

    @sp_property(units="eV")
    def energy(self) -> typing.Tuple[float, float]:
        r0, r1 = self.reactants
        p0, p1 = self.products

        m_r0 = atoms[r0].mass
        m_r1 = atoms[r1].mass
        m_p0 = atoms[p0].mass
        m_p1 = atoms[p1].mass

        f_energy = (m_r0 + m_r1 - m_p0 - m_p1) * scipy.constants.c**2 / scipy.constants.electron_volt

        return f_energy * m_p1 / (m_p0 + m_p1), f_energy * m_p0 / (m_p0 + m_p1)


class NuclearReaction(Dict[Reaction]):
    def get(self, key, default_value=_not_found_) -> Reaction:
        return self._find_(key, default_value=default_value)


def reactivities_DT(ti):
    # H.-S. Bosch and G.M. Hale, Nucl. Fusion 32 (1992) 611.

    # Table VII:
    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.3500e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5
    bg = 34.3827
    er = 1.124656e6
    ti = ti * 1.0e-3
    # Eq. (12)
    r0 = ti * (c2 + ti * (c4 + ti * c6)) / (1.0 + ti * (c3 + ti * (c5 + ti * c7)))
    theta = ti / (1.0 - r0)
    xi = (bg**2 / (4.0 * theta)) ** (1.0 / 3.0)

    sigv = c1 * theta * np.sqrt(xi / (er * (ti) ** 3)) * np.exp(-3.0 * xi)
    return sigv * 1.0e-6  # m^3/s


nuclear_reaction = NuclearReaction(
    {
        r"D(t,n)alpha": {
            "reactants": ["D", "T"],
            "products": ["n", "alpha"],
            "reactivities": reactivities_DT,
            # (
            #     # eV
            #     np.array(
            #         [
            #             0.10e3,
            #             0.20e3,
            #             0.30e3,
            #             0.40e3,
            #             0.50e3,
            #             0.60e3,
            #             0.70e3,
            #             0.80e3,
            #             1.00e3,
            #             1.25e3,
            #             1.30e3,
            #             1.50e3,
            #             1.75e3,
            #             1.80e3,
            #             2.00e3,
            #             2.50e3,
            #             3.00e3,
            #             4.00e3,
            #             5.00e3,
            #             6.00e3,
            #             8.00e3,
            #             10.0e3,
            #             12.0e3,
            #             15.0e3,
            #             20.0e3,
            #             30.0e3,
            #             40.0e3,
            #             50.0e3,
            #         ]
            #     ),
            #     # m^3/s
            #     np.array(
            #         [
            #             0.000e-33,
            #             1.254e-32,
            #             7.292e-31,
            #             9.344e-30,
            #             5.697e-29,
            #             2.253e-28,
            #             6.740e-28,
            #             1.662e-27,
            #             6.857e-27,
            #             2.546e-26,
            #             3.174e-26,
            #             6.923e-26,
            #             1.539e-25,
            #             1.773e-25,
            #             2.977e-25,
            #             8.425e-25,
            #             1.867e-24,
            #             5.974e-24,
            #             1.366e-23,
            #             2.554e-23,
            #             6.222e-23,
            #             1.136e-22,
            #             1.747e-22,
            #             2.740e-22,
            #             4.330e-22,
            #             6.681e-22,
            #             7.998e-22,
            #             8.649e-22,
            #         ]
            #     ),
            # ),
        },
        r"3He(t,n)alpha": {
            "reactants": ["3He", "T"],
            "products": ["n", "alpha"],
            "energy": np.nan,  # eV
            "reactivities": (
                np.array(
                    [
                        0.20e3,
                        0.30e3,
                        0.40e3,
                        0.50e3,
                        0.60e3,
                        0.70e3,
                        0.80e3,
                        1.00e3,
                        1.25e3,
                        1.30e3,
                        1.50e3,
                        1.75e3,
                        1.80e3,
                        2.00e3,
                        2.50e3,
                        3.00e3,
                        4.00e3,
                        5.00e3,
                        6.00e3,
                        8.00e3,
                        10.0e3,
                        12.0e3,
                        15.0e3,
                        20.0e3,
                        30.0e3,
                        40.0e3,
                        50.0e3,
                    ]
                ),
                np.array(
                    [
                        1.414e-41,
                        1.033e-38,
                        6.537e-37,
                        1.241e-35,
                        1.166e-34,
                        6.960e-34,
                        3.032e-33,
                        3.057e-32,
                        2.590e-31,
                        3.708e-31,
                        1.317e-30,
                        4.813e-30,
                        6.053e-30,
                        1.399e-29,
                        7.477e-29,
                        2.676e-28,
                        1.710e-27,
                        6.377e-27,
                        1.739e-26,
                        7.504e-26,
                        2.126e-25,
                        4.715e-25,
                        1.175e-24,
                        3.482e-24,
                        1.363e-23,
                        3.160e-23,
                        5.554e-23,
                    ]
                ),
            ),
        },
        r"D(d,p)T": {
            "reactants": ["D", "D"],
            "products": ["p", "T]"],
            "energy": np.nan,
            "reactivities": (
                np.array(
                    [
                        0.20e3,
                        0.30e3,
                        0.40e3,
                        0.50e3,
                        0.60e3,
                        0.70e3,
                        0.80e3,
                        1.00e3,
                        1.25e3,
                        1.30e3,
                        1.50e3,
                        1.75e3,
                        1.80e3,
                        2.00e3,
                        2.50e3,
                        3.00e3,
                        4.00e3,
                        5.00e3,
                        6.00e3,
                        8.00e3,
                        10.0e3,
                        12.0e3,
                        15.0e3,
                        20.0e3,
                        30.0e3,
                        40.0e3,
                        50.0e3,
                    ]
                ),
                np.array(
                    [
                        4.640e-34,
                        2.071e-32,
                        2.237e-31,
                        1.204e-30,
                        4.321e-30,
                        1.193e-29,
                        2.751e-29,
                        1.017e-28,
                        3.387e-28,
                        4.143e-28,
                        8.431e-28,
                        1.739e-27,
                        1.976e-27,
                        3.150e-27,
                        7.969e-27,
                        1.608e-26,
                        4.428e-26,
                        9.024e-26,
                        1.545e-25,
                        3.354e-25,
                        5.781e-25,
                        8.723e-25,
                        1.390e-24,
                        2.399e-24,
                        4.728e-24,
                        7.249e-24,
                        9.838e-24,
                    ]
                ),
            ),
        },  # eV
        r"D(d,p)^3He": {
            "reactants": ["D", "D"],
            "products": ["p", "3He]"],
            "energy": np.nan,
            "reactivities": (
                np.array(
                    [
                        0.20e3,
                        0.30e3,
                        0.40e3,
                        0.50e3,
                        0.60e3,
                        0.70e3,
                        0.80e3,
                        1.00e3,
                        1.25e3,
                        1.30e3,
                        1.50e3,
                        1.75e3,
                        1.80e3,
                        2.00e3,
                        2.50e3,
                        3.00e3,
                        4.00e3,
                        5.00e3,
                        6.00e3,
                        8.00e3,
                        10.0e3,
                        12.0e3,
                        15.0e3,
                        20.0e3,
                        30.0e3,
                        40.0e3,
                        50.0e3,
                    ]
                ),
                np.array(
                    [
                        4.482e-34,
                        2.004e-32,
                        2.168e-31,
                        1.169e-30,
                        4.200e-30,
                        1.162e-29,
                        2.681e-29,
                        9.933e-29,
                        3.319e-28,
                        4.660e-28,
                        8.284e-28,
                        1.713e-27,
                        1.948e-27,
                        3.110e-27,
                        7.905e-27,
                        1.602e-26,
                        4.447e-26,
                        9.128e-26,
                        1.573e-25,
                        3.457e-25,
                        6.023e-25,
                        9.175e-25,
                        1.481e-24,
                        2.603e-24,
                        5.271e-24,
                        8.235e-24,
                        1.133e-23,
                    ]
                ),
            ),
        },
    }
)

thermal_reactivities = np.array(
    [
        # | $T_i \\ (eV)$ | $D(t,n)\alpha \\ (m^3/s)$ | $^3He(d,p)\alpha \\ (m^3)/s $ | $D(d,p)T \\ (m^3)/s $ | $D(d,p)^3He \\ (m^3)/s $ |
        [0.20e3, 1.254e-32, 1.414e-41, 4.640e-34, 4.482e-34],
        [0.30e3, 7.292e-31, 1.033e-38, 2.071e-32, 2.004e-32],
        [0.40e3, 9.344e-30, 6.537e-37, 2.237e-31, 2.168e-31],
        [0.50e3, 5.697e-29, 1.241e-35, 1.204e-30, 1.169e-30],
        [0.60e3, 2.253e-28, 1.166e-34, 4.321e-30, 4.200e-30],
        [0.70e3, 6.740e-28, 6.960e-34, 1.193e-29, 1.162e-29],
        [0.80e3, 1.662e-27, 3.032e-33, 2.751e-29, 2.681e-29],
        [1.00e3, 6.857e-27, 3.057e-32, 1.017e-28, 9.933e-29],
        [1.25e3, 2.546e-26, 2.590e-31, 3.387e-28, 3.319e-28],
        [1.30e3, 3.174e-26, 3.708e-31, 4.143e-28, 4.660e-28],
        [1.50e3, 6.923e-26, 1.317e-30, 8.431e-28, 8.284e-28],
        [1.75e3, 1.539e-25, 4.813e-30, 1.739e-27, 1.713e-27],
        [1.80e3, 1.773e-25, 6.053e-30, 1.976e-27, 1.948e-27],
        [2.00e3, 2.977e-25, 1.399e-29, 3.150e-27, 3.110e-27],
        [2.50e3, 8.425e-25, 7.477e-29, 7.969e-27, 7.905e-27],
        [3.00e3, 1.867e-24, 2.676e-28, 1.608e-26, 1.602e-26],
        [4.00e3, 5.974e-24, 1.710e-27, 4.428e-26, 4.447e-26],
        [5.00e3, 1.366e-23, 6.377e-27, 9.024e-26, 9.128e-26],
        [6.00e3, 2.554e-23, 1.739e-26, 1.545e-25, 1.573e-25],
        [8.00e3, 6.222e-23, 7.504e-26, 3.354e-25, 3.457e-25],
        [10.0e3, 1.136e-22, 2.126e-25, 5.781e-25, 6.023e-25],
        [12.0e3, 1.747e-22, 4.715e-25, 8.723e-25, 9.175e-25],
        [15.0e3, 2.740e-22, 1.175e-24, 1.390e-24, 1.481e-24],
        [20.0e3, 4.330e-22, 3.482e-24, 2.399e-24, 2.603e-24],
        [30.0e3, 6.681e-22, 1.363e-23, 4.728e-24, 5.271e-24],
        [40.0e3, 7.998e-22, 3.160e-23, 7.249e-24, 8.235e-24],
        [50.0e3, 8.649e-22, 5.554e-23, 9.838e-24, 1.133e-23],
    ]
)
