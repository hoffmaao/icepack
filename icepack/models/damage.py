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
from firedrake import sym, inner, grad, dot, div, det, tr, Identity, sqrt, dx, ds, dS
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
    year,
)
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.utilities import add_kwarg_wrapper


def get_eig(hes):
    mesh = hes.function_space().mesh()
    [eigL, eigR] = np.linalg.eig(
        hes.vector().array().reshape([mesh.num_vertices(), 2, 2])
    )
    eig = firedrake.Function(VectorFunctionSpace(mesh, "CG", 1))
    eig.vector().set_local(eigL.flatten())
    return eig


def M(eps, A):
    I = Identity(2)
    trace = tr(eps)
    eps_e = sqrt((inner(eps, eps) + trace ** 2) / 2)
    mu = 0.5 * A ** (-1 / n) * eps_e ** (1 / n - 1)
    return 2 * mu * (eps + trace * I)


def M_e(eps_e, A):
    return sqrt(3.0) * A ** (-1 / n) * eps_e ** (1 / n)


def heal(e1, eps_h, dt, lh=0.1):
    return dt * lh * (e1 - eps_h)


def fracture(D, eps_e, dt, ld=0.3):
    return dt * ld * eps_e * (1 - D)


class DamageTransport(object):
    def solve(self, dt, D0, u, D_inflow=None, **kwargs):
        """Propogate the ice damage forward in time by one timestep
      This function uses a Runge-Kutta scheme to upwind damage 
      (limiting damage diffusion) while sourcing and sinking 
      damage assocaited with crevasse opening/crevasse healing
    Parameters
    ----------
    dt : float
      timestep
    D0 : firedrake.Function
      initial damage feild should be discontinuous
    u : firedrake.Function
      ice velocity
    D_inflow : firedrake.Function
      damage of the upstream ice that advects into the domain
    Returns
    D : firedrake.Function
      advected ice damage at `t + dt`
    """

        D_inflow = D_inflow if D_inflow is not None else D0
        Q = D0.function_space()
        dD, ϕ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        d = ϕ * dD * dx
        D = D0.copy(deepcopy=True)

        """ unit normal for facets in mesh, Q """
        n = firedrake.FacetNormal(Q.mesh())

        """ find the upstream direction and solve
            for advected damage """
        un = 0.5 * (dot(u, n) + abs(dot(u, n)))
        L1 = dt * (
            D * div(ϕ * u) * dx
            - firedrake.conditional(dot(u, n) < 0, ϕ * dot(u, n) * D_inflow, 0.0) * ds
            - firedrake.conditional(dot(u, n) > 0, ϕ * dot(u, n) * D, 0.0) * ds
            - (ϕ("+") - ϕ("-")) * (un("+") * D("+") - un("-") * D("-")) * dS
        )
        D1 = firedrake.Function(Q)
        D2 = firedrake.Function(Q)
        L2 = firedrake.replace(L1, {D: D1})
        L3 = firedrake.replace(L1, {D: D2})

        dq = firedrake.Function(Q)

        """ three-stage strong-stability-preserving Runge-Kutta 
            (SSPRK) scheme for advecting damage """
        params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
        prob1 = firedrake.LinearVariationalProblem(d, L1, dq)
        solv1 = firedrake.LinearVariationalSolver(prob1, solver_parameters=params)
        prob2 = firedrake.LinearVariationalProblem(d, L2, dq)
        solv2 = firedrake.LinearVariationalSolver(prob2, solver_parameters=params)
        prob3 = firedrake.LinearVariationalProblem(d, L3, dq)
        solv3 = firedrake.LinearVariationalSolver(prob3, solver_parameters=params)

        solv1.solve()
        D1.assign(D + dq)
        solv2.solve()
        D2.assign(0.75 * D + 0.25 * (D1 + dq))
        solv3.solve()
        D.assign((1.0 / 3.0) * D + (2.0 / 3.0) * (D2 + dq))

        return D


class Levermann(object):
    def solve(self, dt, D0, u, A, ld=0.3, lh=0.1, D_max=1.0, **kwargs):
        """ Damage advected, solve for stress and add new damage
            for von mises criterion σc = 3.0^0.5*B*εdot**(1/n).
        for maximum shear stress criterion (Tresca or Guest criterion) 
            σs = max(|σl|, |σt|,|σl-σt|)
        ----------
        dt : float
            timestep
        D0 : firedrake.Function
            initial damage feild should be discontinuous
        u : firedrake.Function
            Ice velocity
        A : firedrake.Function
            fluidity parameter
        ld : float
            damage source coefficient
        lh : float
            damage healing coefficient
        D_max : fload
            maximum damage
        Returns
        Dlevermann : firedrake.Function
            damage after ice fracture healing
        """
        Q = D0.function_space()
        D = D0.copy(deepcopy=True)
        h_term = firedrake.Function(Q)
        f_term = firedrake.Function(Q)
        Dlevermann = firedrake.Function(Q)

        eps = sym(grad(u))
        trace_e = tr(eps)
        det_e = det(eps)
        eig = [
            1 / 2 * trace_e + sqrt(trace_e ** 2 - 4 * det_e),
            1 / 2 * trace_e - sqrt(trace_e ** 2 - 4 * det_e),
        ]
        e1 = firedrake.max_value(*eig)
        e2 = firedrake.min_value(*eig)
        eps_e = sqrt((inner(eps, eps) + trace_e ** 2) / 2)

        σ = M(eps, A)
        σc = M_e(eps_e, A)
        trace_s = tr(σ)
        σ_e = sqrt((inner(σ, σ) + trace_s ** 2) / 2)
        eps_h = 2.0 * 10 ** -10 * year

        """ add damage associated with longitudinal spreading after 
        advecting damage feild.  """
        h_term.project(
            firedrake.conditional(e1 - eps_h < 0, heal(e1, eps_h, dt, lh), 0.0)
        )
        f_term.project(
            firedrake.conditional(σ_e - σc > 0, fracture(D, eps_e, dt, ld), 0.0)
        )

        """ we require that damage be in the set [0,D_max) """
        Dlevermann.project(
            firedrake.conditional(
                D + f_term + h_term > D_max, D_max, D + f_term + h_term
            )
        )
        Dlevermann.project(
            firedrake.conditional(D + f_term + h_term < 0.0, 0.0, D + f_term + h_term)
        )

        return Dlevermann


class Damage(object):
    r"""Class for modelling ice damage

    This class provides functions that solve for ice damage.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """

    def __init__(
        self,
        viscosity=viscosity,
        damage_transport=DamageTransport(),
        damage=Levermann(),
    ):

        self.viscosity = add_kwarg_wrapper(viscosity)
        self.damage_transport = damage_transport
        self.damage = damage
