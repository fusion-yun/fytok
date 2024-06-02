import typing

import numpy as np
import dataclasses
import collections.abc
import scipy.interpolate
from skimage import measure

from spdm.core.field import Field
from spdm.core.expression import Variable
from spdm.geometry.curve import Curve
from spdm.core.geo_object import GeoObject
from spdm.geometry.point import Point
from spdm.utils.logger import deprecated, logger
from spdm.utils.tags import _not_found_

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

        if len(data) == 0:
            yield None
        elif data.shape[0] == 1:
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
        xy = f.mesh.points
        x = xy[0]
        y = xy[1]
        z = np.asarray(f)
    else:
        raise ValueError(f"Wrong number of arguments! should be 1 or 3, got {len(args)}")

    yield from find_countours_skimage(values, z, x, y, **kwargs)


@dataclasses.dataclass
class OXPoint:
    r: float
    z: float
    value: float


def find_critical_points(psi: Field) -> typing.Tuple[typing.Sequence[OXPoint], typing.Sequence[OXPoint]]:
    opoints = []

    xpoints = []

    R, Z = psi.mesh.points
    _R = Variable(0, "R")
    Bp2 = (psi.pd(0, 1) ** 2 + psi.pd(1, 0) ** 2) / (_R**2)

    D = psi.pd(2, 0) * psi.pd(0, 2) - psi.pd(1, 1) ** 2

    for r, x_z in minimize_filter(Bp2, R, Z):
        p = OXPoint(r, x_z, psi(r, x_z))

        if D(r, x_z) < 0.0:  # saddle/X-point
            xpoints.append(p)
        else:  # extremum/ O-point
            opoints.append(p)

    Rmid, Zmid = psi.mesh.geometry.bbox.origin + psi.mesh.geometry.bbox.dimensions * 0.5

    opoints.sort(key=lambda x: (x.r - Rmid) ** 2 + (x.z - Zmid) ** 2)

    # TODO:

    o_psi = opoints[0].value
    o_r = opoints[0].r
    o_z = opoints[0].z

    # remove illegal x-points . learn from freegs
    # check psi should be monotonic from o-point to x-point

    x_points = []
    s_points = []
    for xp in xpoints:
        length = 20

        psiline = psi(np.linspace(o_r, xp.r, length), np.linspace(o_z, xp.z, length))

        if len(np.unique(psiline[1:] > psiline[:-1])) != 1:
            s_points.append(xp)
        else:
            x_points.append(xp)

    xpoints = x_points

    xpoints.sort(key=lambda x: (x.value - o_psi) ** 2)

    if len(opoints) == 0 or len(xpoints) == 0:
        raise RuntimeError(f"Can not find O-point or X-point! {opoints} {xpoints}")

    return opoints, xpoints


def find_contours_old(psirz: Field, psi, axis=None) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
    """
    if axis is not None:
        only return  closed surface  enclosed axis
        if closed surface does not exists, return None
        number of surface == len(psi)
    else:
        do not guarantee the number of surface == len(psi)
        return all surface ,
    """
    if psi is None or psi is _not_found_:
        psi = []
    elif isinstance(psi, float):
        psi = [psi]

    if axis is None or axis is False:
        for psi_val, surfs in find_contours(psirz, values=psi):
            for surf in surfs:
                if isinstance(surf, GeoObject):
                    surf.set_coordinates("r", "z")
                yield psi_val, surf

    else:
        # x_point = None
        if isinstance(axis, OXPoint):
            o_point = axis
        elif isinstance(axis, (list, tuple)):
            o_point = OXPoint(*axis)
        else:
            raise TypeError(f"Wrong type of argument! should be OXPoint, list or tuple, got {type(axis)}")

        if isinstance(psi, float):
            psi = [psi]

        current_psi = np.nan
        current_count = 0
        for psi_val, surfs in find_contours(psirz, values=psi):
            # 累计相同 level 的 surface个数
            # 如果累计的 surface 个数大于1，说明存在磁岛
            # 如果累计的 surface 个数等于0，说明该 level 对应的 surface 不存在
            # 如果累计的 surface 个数等于1，说明该 level 对应的 surface 存在且唯一

            count = 0
            for surf in surfs:
                count += 1
                if surf is None and np.isclose(psi_val, o_point.value):
                    yield psi_val, Point(o_point.r, o_point.z)
                elif isinstance(surf, Point) and all(np.isclose(surf.points, [o_point.r, o_point.z])):
                    yield psi_val, surf  # raise RuntimeError(f"Can not find surface psi={level}")
                elif isinstance(surf, Curve) and surf.is_closed and surf.enclose(o_point.r, o_point.z):
                    # theta_0 = np.arctan2(x_point.r-o_point.r, x_point.z-o_point.z)
                    # theta = ((np.arctan2(_R-o_point.r, _Z-o_point.z)-theta_0)+2.0*scipy.constants.pi) % (2.0*scipy.constants.pi)
                    # surf = surf.remesh(theta)
                    surf.set_coordinates("r", "z")
                    yield psi_val, surf
                else:
                    count -= 1
            if count <= 0:
                if np.isclose(psi_val, o_point.value):
                    yield psi_val, Point(o_point.r, o_point.z)
                else:
                    # logger.warning(f"{psi_val} {o_point.psi}")
                    yield psi_val, None
            elif current_count > 1:
                raise RuntimeError(f"Something wrong! Get {current_count} closed surfaces for psi={current_psi}")

            # theta = np.arctan2(surf[:, 0]-o_point.r, surf[:, 1]-o_point.z)
            # logger.debug((max(theta)-min(theta))/(2.0*scipy.constants.pi))
            # if 1.0 - (max(theta)-min(theta))/(2.0*scipy.constants.pi) > 2.0/len(theta):  # open or do not contain o-point
            #     current_count -= 1
            #     continue

            # is_closed = False

            # if np.isclose((theta[0]-theta[-1]) % (2.0*scipy.constants.pi), 0.0):
            #     # 封闭曲线
            #     theta = theta[:-1]
            #     surf = surf[:-1]
            #     # is_closed = True
            # else:  # boundary separatrix
            #     if x_point is None:
            #         raise RuntimeError(f"No X-point ")
            #     # logger.warning(f"The magnetic surface average is not well defined on the separatrix!")
            #     xpt = np.asarray([x_point.r, x_point.z], dtype=float)
            #     b = surf[1:]
            #     a = surf[:-1]
            #     d = b-a
            #     d2 = d[:, 0]**2+d[:, 1]**2
            #     p = xpt-a

            #     c = (p[:, 0]*d[:, 0]+p[:, 1]*d[:, 1])/d2
            #     s = (p[:, 0]*d[:, 1]-p[:, 1]*d[:, 0])/d2
            #     idx = np.flatnonzero(np.logical_and(c >= 0, c**2+s**2 < 1))

            #     if len(idx) == 2:

            #         idx0 = idx[0]
            #         idx1 = idx[1]

            #         theta_x = np.arctan2(xpt[0]-o_point.r, xpt[1]-o_point.z)

            #         surf = np.vstack([[xpt], surf[idx0:idx1]])
            #         theta = np.hstack([theta_x, theta[idx0:idx1]])
            #     else:
            #         raise RuntimeError(f"Can not get closed boundary {o_point}, {x_point} {idx} !")

            # # theta must be strictly increased
            # p_min = np.argmin(theta)
            # p_max = np.argmax(theta)

            # if p_min > 0:
            #     if p_min == p_max+1:
            #         theta = np.roll(theta, -p_min)
            #         surf = np.roll(surf, -p_min, axis=0)
            #     elif p_min == p_max-1:
            #         theta = np.flip(np.roll(theta, -p_min-1))
            #         surf = np.flip(np.roll(surf, -p_min-1, axis=0), axis=0)
            #     else:
            #         raise ValueError(f"Can not convert 'u' to be strictly increased!")
            #     theta = np.hstack([theta, [theta[0]+(2.0*scipy.constants.pi)]])
            #     theta = (theta-theta.min())/(theta.max()-theta.min())
            #     surf = np.vstack([surf, surf[:1]])

            # if surf.shape[0] == 0:
            #     logger.warning(f"{level},{o_point.psi},{(max(theta),min(theta))}")

            # elif surf.shape[0] == 1:
            #     yield level, Point(surf[0][0], surf[0][1])
            # else:
            #     yield level, Curve(surf, theta, is_closed=is_closed)
