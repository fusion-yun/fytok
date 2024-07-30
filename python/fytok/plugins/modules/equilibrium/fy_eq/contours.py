import typing
import numpy as np
import collections.abc
import scipy.interpolate
from skimage import measure


from spdm.core.field import Field
from spdm.core.expression import Variable
from spdm.core.htree import List
from spdm.core.geo_object import GeoObject

from spdm.geometry.curve import Curve
from spdm.geometry.point import Point


from .optimize import minimize_filter

# import matplotlib.pyplot as plt
# @deprecated
# def find_contours_matplotlib(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, /, *args, levels=None, ** kwargs) -> typing.List[typing.List[np.ndarray]]:
#     """
#         args:X: np.ndarray, Y: np.ndarray, Z: np.ndarray
#         TODO: need improvement
#     """
#     fig = plt.figure()
#     contour_set = fig.gca().contour(x, y, z, *args, levels=levels, ** kwargs)
#     return [(contour_set.levels[idx], col.get_segments()) for idx, col in enumerate(contour_set.collections)]


def find_countours_skimage_(
    val: float, z: np.ndarray, x_inter, y_inter
) -> typing.Generator[GeoObject | None, None, None]:
    for c in measure.find_contours(z, val):
        # data = [[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c]
        x = np.asarray(x_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        y = np.asarray(y_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        data = np.stack([x, y], axis=-1)

        if data.shape[0] == 1:
            yield Point(*data[0])
        else:
            yield Curve(data)


def find_countours_skimage(vals: list, z: np.ndarray, x: np.ndarray, y: np.ndarray):
    if z.shape == x.shape and z.shape == y.shape:
        pass
    else:
        raise ValueError(f"Array shape does not match! x:{x.shape} , y:{y.shape}, z:{z.shape} ")
    shape = z.shape
    dim0 = np.linspace(0, shape[0] - 1, shape[0])
    dim1 = np.linspace(0, shape[1] - 1, shape[1])
    x_inter = scipy.interpolate.RectBivariateSpline(dim0, dim1, x)
    y_inter = scipy.interpolate.RectBivariateSpline(dim0, dim1, y)

    if not isinstance(vals, (collections.abc.Sequence, np.ndarray)):
        vals = [vals]
    elif isinstance(vals, np.ndarray) and vals.ndim == 0:
        vals = vals.reshape([1])

    for val in vals:
        yield val, find_countours_skimage_(val, z, x_inter, y_inter)

        # count = 0
        # for c in measure.find_contours(z, val):
        #     count += 1
        #     # data = [[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c]
        #     x = np.asarray(x_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        #     y = np.asarray(y_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        #     data = np.stack([x, y], axis=-1)
        #     if len(data) == 0:
        #         yield val, None
        #     elif data.shape[0] == 1:
        #         yield val, Point(*data[0])
        #     else:
        #         yield val, Curve(data)
        # if count == 0:
        #     yield val, None


def find_contours(
    *args, values, **kwargs
) -> typing.Generator[typing.Tuple[float, typing.Generator[GeoObject | None, None, None]], None, None]:
    if len(args) == 3:
        z, x, y = args
    elif len(args) == 1:
        if not isinstance(args[0], Field):
            raise TypeError(f"Wrong type of argument! should be Field, got {type(args[0])}")
        f = args[0]
        x, y = f.mesh.coordinates
        z = np.asarray(f)
    else:
        raise ValueError(f"Wrong number of arguments! should be 1 or 3, got {len(args)}")

    yield from find_countours_skimage(values, z, x, y, **kwargs)


def find_critical_points(
    psi: Field,
) -> typing.Tuple[List[typing.Tuple[Point, float]], List[typing.Tuple[Point, float]]]:
    opoints = []

    xpoints = []

    R, Z = psi.mesh.coordinates
    _R = Variable(0, "R")
    Bp2 = (psi.pd(0, 1) ** 2 + psi.pd(1, 0) ** 2) / (_R**2)

    D = psi.pd(2, 0) * psi.pd(0, 2) - psi.pd(1, 1) ** 2

    for r, z in minimize_filter(Bp2, R, Z):
        pv = (Point(r, z), psi(r, z))

        if D(r, z) < 0.0:  # saddle/X-point
            xpoints.append(pv)
        else:  # extremum/ O-point
            opoints.append(pv)

    Rmid, Zmid = psi.mesh.geometry.bbox.origin + psi.mesh.geometry.bbox.dimensions * 0.5

    opoints.sort(key=lambda x: (x[0][0] - Rmid) ** 2 + (x[0][1] - Zmid) ** 2)

    # TODO:

    o_psi = opoints[0][1]
    o_r = opoints[0][0][0]
    o_z = opoints[0][0][1]

    # remove illegal x-points . learn from freegs
    # check psi should be monotonic from o-point to x-point

    x_points = []
    s_points = []
    for xp in xpoints:
        length = 20

        psiline = psi(
            np.linspace(o_r, xp[0][0], length),
            np.linspace(o_z, xp[0][1], length),
        )

        if len(np.unique(psiline[1:] > psiline[:-1])) != 1:
            s_points.append(xp)
        else:
            x_points.append(xp)

    xpoints = x_points

    xpoints.sort(key=lambda x: (x[1] - o_psi) ** 2)

    if len(opoints) == 0 or len(xpoints) == 0:
        raise RuntimeError(f"Can not find O-point or X-point! {opoints} {xpoints}")

    return opoints, xpoints
