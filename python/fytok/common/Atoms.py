from spdm.numlib import constants

atoms = {
    "e": {
        "label": "e",
        "z": -1,
        "element": [{"a": constants.m_e/constants.m_p, "z_n": 1, "atoms_n": 1}],
    },
    "H": {
        "label": "H",
        "z": 1,
        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],

    },
    "D": {
        "label": "D",
        "z": 1,
        "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],

    },
    "T": {
        "label": "T",
        "z": 1,
        "element": [{"a": 3, "z_n": 1, "atoms_n": 1}],

    },
    "He": {
        "label": "He",
        "z": 2,
        "element": [{"a": 4, "z_n": 1, "atoms_n": 1}],

    },
    "Be": {
        "label": "Be",
        "z": 4,
        "element": [{"a": 9, "z_n": 1, "atoms_n":   1}],

    },
    "Ar": {
        "label": "Ar",
        "z": 18,
        "element": [{"a": 40, "z_n": 1, "atoms_n":   1}],

    }
}
