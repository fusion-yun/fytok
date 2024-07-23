import collections.abc
import os
import typing
import numpy as np
import scipy.optimize
import scipy.ndimage  # for maximum_filter,binary_erosion, generate_binary_structure

from spdm.core.expression import Variable
from spdm.core.field import Field
from spdm.utils.logger import logger
from spdm.utils.type_hint import ArrayType, NumericType, ScalarType

SP_EXPERIMENTAL = os.environ.get("SP_EXPERIMENTAL", False)

# logger.info(f"SP_EXPERIMENTAL \t: {SP_EXPERIMENTAL}")

EPSILON = 1.0e-2


def minimize_filter(
    func: typing.Callable[..., ScalarType | ArrayType],
    X,
    Y,
    width=None,
    tolerance: float = None,
    method="L-BFGS-B",
    # xmin: float, ymin: float, xmax: float, ymax: float, tolerance: float = EPSILON
):
    # if isinstance(tolerance, float):
    #     dx = tolerance
    #     dy = tolerance
    # elif isinstance(tolerance, (collections.abc.Sequence, np.ndarray)) and len(tolerance) == 2:
    #     dx, dy = tolerance
    # else:
    #     raise TypeError(f"Illegal type {type(dx)}")

    # nx = int((xmax-xmin)/dx)+1
    # ny = int((ymax-ymin)/dy)+1

    # X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
    #                    np.linspace(ymin, ymax, ny), indexing='ij')
    data = func(X, Y)

    nx, ny = data.shape

    z_max = np.max(data)

    z_min = np.min(data)

    if tolerance is None:
        tolerance = np.abs(z_max - z_min) / np.sqrt(nx * ny) * 2
        if tolerance > 0.01 or tolerance < 0.001:
            tolerance = 0.01

    if isinstance(width, collections.abc.Sequence):
        wx, wy = width
    else:
        wx = int(max(4, nx / 32))
        wy = int(max(4, ny / 32))

    peak = scipy.ndimage.minimum_filter(data, size=(wx, wy), mode="constant") == data

    idxs = np.asarray(np.where(peak)).T

    for ix, iy in idxs:
        if ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1:
            continue

        r = np.abs((data[ix, iy] - z_min) / (z_max - z_min))

        if r > tolerance:
            continue

        xmin = X[ix - 1, iy]
        xmax = X[ix + 1, iy]
        ymin = Y[ix, iy - 1]
        ymax = Y[ix, iy + 1]

        x = X[ix, iy]
        y = Y[ix, iy]

        # if True:

        sol = scipy.optimize.minimize(
            lambda x: func(x[0], x[1]),
            np.asarray([x, y]),
            bounds=[(xmin, xmax), (ymin, ymax)],
            method=method,
            tol=tolerance,
        )

        xsol, ysol = sol.x

        if not (x >= xmin or x <= xmax or y >= ymin or y <= ymax):
            continue
        elif sol.success:
            yield xsol, ysol
        else:
            logger.warning(f"{sol.message} at {xsol, ysol} ")
            yield x, y
