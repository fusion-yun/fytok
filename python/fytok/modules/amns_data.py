import numpy as np

from spdm.core.htree import Dict, HTree
from spdm.core.expression import Expression
from spdm.core.function import Function, Polynomials
from spdm.core.sp_tree import sp_property, sp_tree, SpTree
from spdm.utils.tags import _not_found_
from spdm.utils.typing import array_type


from ..ontology import amns_data


class AMNSProcess(SpTree):
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (list, array_type)):
            args = [{"radiation": args[0]}, *args[1:]]
        super().__init__(*args, **kwargs)

    radiation: Polynomials = sp_property(
        units="eV*m^3/s",
        type="chebyshev",
        preprocess=(lambda x: -1.0 + 2 * np.log(np.abs(x) / 50) / np.log(1000)),  # domain 50eV ~ 50000eV
        postprocess=(lambda y: np.exp(y) / (1.6022e-12 * 1.0e6)),  # change units erg.cm^-3/s => eV.m^-3/s
    )


class AMNS(Dict[AMNSProcess]):
    def _find_(self, key: str, *args, **kwargs) -> AMNSProcess:
        _key = key
        while isinstance(_key, str):
            res = self.get_cache(_key, _not_found_)
            if isinstance(res, str):
                _key = res
            else:
                break

        if res is _not_found_:
            res = self.__missing__(key)

        if not isinstance(res, AMNSProcess):
            res = self._type_convert(res, key, _type_hint=AMNSProcess)

        return res


############################################################################
# copy from gacode/tgyro_rad.f90 , should be replaced by AMNS or other Open ADAS tools
amns = AMNS(
    {  # Chebyshev expansion coefficients: c_n
        "W": [
            -4.093426327035e01,
            -8.887660631564e-01,
            -3.780990284830e-01,
            -1.950023337795e-01,
            +3.138290691843e-01,
            +4.782989513315e-02,
            -9.942946187466e-02,
            +8.845089763161e-03,
            +9.069526573697e-02,
            -5.245048352825e-02,
            -1.487683353273e-02,
            +1.917578018825e-02,
        ],
        "Xe": [
            -4.126366679797e01,
            -1.789569183388e00,
            -2.380331458294e-01,
            +2.916911530426e-01,
            -6.217313390606e-02,
            +1.177929596352e-01,
            +3.114580325620e-02,
            -3.551020007260e-02,
            -4.850122964780e-03,
            +1.132323304719e-02,
            -5.275312157892e-02,
            -9.051568201374e-03,
        ],
        "Mo": [
            -4.178151951275e01,
            -1.977018529373e00,
            +5.339155696054e-02,
            +1.164267551804e-01,
            +3.697881990263e-01,
            -9.594816048640e-02,
            -1.392054581553e-01,
            +1.272648056277e-01,
            -1.336366483240e-01,
            +3.666060293888e-02,
            +9.586025795242e-02,
            -7.210209944439e-02,
        ],
        "Kr": [
            -4.235332287815e01,
            -1.508707679199e00,
            -3.300772886398e-01,
            +6.166385849657e-01,
            +1.752687990068e-02,
            -1.004626261246e-01,
            +5.175682671490e-03,
            -1.275380183939e-01,
            +1.087790584052e-01,
            +6.846942959545e-02,
            -5.558980841419e-02,
            -6.669294912560e-02,
        ],
        "Ni": [
            -4.269403899818e01,
            -2.138567547684e00,
            +4.165648766103e-01,
            +2.507972619622e-01,
            -1.454986877598e-01,
            +4.044612562765e-02,
            -1.231313167536e-01,
            +1.307076922327e-01,
            +1.176971646853e-01,
            -1.997449027896e-01,
            -8.027057678386e-03,
            +1.583614529900e-01,
        ],
        "Fe": [
            -4.277490044241e01,
            -2.232798257858e00,
            +2.871183684045e-01,
            +2.903760139426e-01,
            -4.662374777924e-02,
            -4.436273974526e-02,
            -1.004882554335e-01,
            +1.794710746088e-01,
            +3.168699330882e-02,
            -1.813266337535e-01,
            +5.762415716395e-02,
            +6.379542965373e-02,
        ],
        "Ca": [
            -4.390083075521e01,
            -1.692920511934e00,
            +1.896825846094e-01,
            +2.333977195162e-01,
            +5.307786998918e-02,
            -2.559420140904e-01,
            +4.733492400000e-01,
            -3.788430571182e-01,
            +3.375702537147e-02,
            +1.030183684347e-01,
            +1.523656115806e-02,
            -7.482021324342e-02,
        ],
        "Ar": [
            -4.412345259739e01,
            -1.788450950589e00,
            +1.322515262175e-01,
            +4.876947389538e-01,
            -2.869002749245e-01,
            +1.699452914498e-01,
            +9.950501421570e-02,
            -2.674585184275e-01,
            +7.451345261250e-02,
            +1.495713760953e-01,
            -1.089524173155e-01,
            -4.191575231760e-02,
        ],
        "Si": [
            -4.459983387390e01,
            -2.279998599897e00,
            +7.703525425589e-01,
            +1.494919348709e-01,
            -1.136851457700e-01,
            +2.767894295326e-01,
            -3.577491771736e-01,
            +7.013841334798e-02,
            +2.151919651291e-01,
            -2.052895326141e-01,
            +2.210085804088e-02,
            +9.270982150548e-02,
        ],
        "Al": [
            -4.475065090279e01,
            -2.455868594007e00,
            +9.468903008039e-01,
            +6.944445017599e-02,
            -4.550919134508e-02,
            +1.804382546971e-01,
            -3.573462505157e-01,
            +2.075274089736e-01,
            +1.024482383310e-01,
            -2.254367207993e-01,
            +1.150695613575e-01,
            +3.414328980459e-02,
        ],
        "Ne": [
            -4.599844680574e01,
            -1.684860164232e00,
            +9.039325377493e-01,
            -7.544604235334e-02,
            +2.849631706915e-01,
            -4.827471944126e-01,
            +3.138177972060e-01,
            +2.876874062690e-03,
            -1.809607030192e-01,
            +1.510609882754e-01,
            -2.475867654255e-02,
            -6.269602018004e-02,
        ],
        "F": [
            -4.595870691474e01,
            -2.176917325041e00,
            +1.176783264877e00,
            -7.712313240060e-02,
            +1.847534287214e-01,
            -4.297192280031e-01,
            +3.374503944631e-01,
            -5.862051731844e-02,
            -1.363051725174e-01,
            +1.580531615737e-01,
            -7.677594113938e-02,
            -5.498186771891e-03,
        ],
        "N": [
            -4.719917668483e01,
            -1.128938430123e00,
            +5.686617156868e-01,
            +5.565647850806e-01,
            -6.103105546858e-01,
            +2.559496676285e-01,
            +3.204394187397e-02,
            -1.347036917773e-01,
            +1.166192946931e-01,
            -6.001774708924e-02,
            +1.078186024405e-02,
            +1.336864982060e-02,
        ],
        "O": [
            -4.688092238361e01,
            -1.045540847894e00,
            +3.574644442831e-01,
            +6.007860794100e-01,
            -3.812470436912e-01,
            -9.944716626912e-02,
            +3.141455586422e-01,
            -2.520592337580e-01,
            +9.745206757309e-02,
            +1.606664371633e-02,
            -5.269687016804e-02,
            +3.726780755484e-02,
        ],
        "C": [
            -4.752370087442e01,
            -1.370806613078e00,
            +1.119762977201e00,
            +6.244262441360e-02,
            -4.172077577493e-01,
            +3.237504483005e-01,
            -1.421660253114e-01,
            +2.526893756273e-02,
            +2.320010310338e-02,
            -3.487271688767e-02,
            +2.758311539699e-02,
            -1.063180164276e-02,
        ],
        "Be": [
            -4.883447566291e01,
            -8.543314577695e-01,
            +1.305444973614e00,
            -4.830394934711e-01,
            +1.005512839480e-01,
            +1.392590190604e-02,
            -1.980609625444e-02,
            +5.342857189984e-03,
            +2.324970825974e-03,
            -2.466382923947e-03,
            +1.073116177574e-03,
            -9.834117466066e-04,
        ],
        "He": [
            -5.128490291648e01,
            +7.743125302555e-01,
            +4.674917416545e-01,
            -2.087203609904e-01,
            +7.996303682551e-02,
            -2.450841492530e-02,
            +4.177032799848e-03,
            +1.109529527611e-03,
            -1.080271138220e-03,
            +1.914061606095e-04,
            +2.501544833223e-04,
            -3.856698155759e-04,
        ],
        # Hydrogen-like ions (H,D,T)
        "H": [
            -5.307012989032e01,
            +1.382271913121e00,
            +1.111772196884e-01,
            -3.989144654893e-02,
            +1.043427394534e-02,
            -3.038480967797e-03,
            +5.851591993347e-04,
            +3.472228652286e-04,
            -8.418918897927e-05,
            +3.973067124523e-05,
            -3.853620366361e-05,
            +2.005063821667e-04,
        ],
        "D": "H",
        "T": "H",
    }
)


if __name__ == "__main__":
    from fytok.modules.amns_data import amns
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(200, 10000)
    for s in ["H", "D", "He"]:
        plt.plot(x, amns[s].radiation(x), label=s)
    plt.ylabel(r"$ev \cdot m^{3}/s$")
    plt.xlabel("$T[eV]$")
    plt.legend()
