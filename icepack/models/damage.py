# Copyright (C) 2017-2019 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

import firedrake
from firedrake import inner, grad, dx, ds
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               gravity as g)
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import side_friction, normal_flow_penalty
from icepack.models.mass_transport import MassTransport
from icepack.models.damage_transport import DamageTransport
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper




class Damage(object):
    r"""Class for modelling ice damage

    This class provides functions that solve for ice damage.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """
    def __init__(self, viscosity=viscosity):
        self.damage_transport = DamageTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)



