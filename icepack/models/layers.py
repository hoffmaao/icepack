# Copyright (C) 2018-2019 by Daniel Shapero <shapero@uw.edu> and Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew Hoffman's development branch of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

"""Solver for internal layers representing isochronal structures
"""

import firedrake
import icepack.models.viscosity
import math
import numpy as np
from firedrake import atan, project
from icepack.constants import year, glen_flow_law as n


def orthogonal_velocity(u, h):
    """Solve for velocity orthogonal to bed
    ----------
    u : firedrake.Function
        Ice velocity
    h : firedrake.Function
        Ice thickness """

    Q = h.function_space()
    w = project(-(u[0].dx(0) + u[1].dx(1)), Q)
    return w


class slopes(object):
    def solve(self, u, h, **kwargs):
        r"""Solve for steady state layer slopes in radians
		Parameters
        ----------
		u : firedrake.Function
			Ice velocity

		h : firedrake.Function
			Ice thickness

		s : firedrake.Function
			Ice surface elevation
		"""

        V = u.function_space()
        w = orthogonal_velocity(u, h)
        slopes = project(firedrake.as_vector((atan(w / u[0]), atan(w / u[1]))), V)

        return slopes
