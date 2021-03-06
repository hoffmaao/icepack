# Copyright (C) 2017-2019 by Daniel Shapero <shapero@uw.edu>
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

r"""Solvers for inverse problems

This module contains objects for specifying and solving inverse problems, where
some unobservable field is estimated based on observations of an observable
field and a physical model that connects the two. The class `InverseProblem` is
used to specify the problem to be solved, while the classes that inherit from
`InverseSolver` are used to solve it.
"""

import numpy as np
import scipy.optimize
import firedrake
from firedrake import action, adjoint, derivative, replace, ln, dx


def _bracket(f):
    r"""Given a decreasing real function of a single variable, return a value
    `t` such that `f(t) < f(0)`, which can then be used for a more thorough
    line search"""
    f_0 = f(0)
    t = 1.0
    while True:
        try:
            f_t = f(t)
            if f_t < f_0:
                return t
        except (AssertionError, firedrake.ConvergenceError):
            pass

        t /= 2


class InverseProblem(object):
    r"""Specifies an inverse problem

    This object is used to specify an inverse problem, i.e. estimating a
    parameter :math:`p` from measurements of a field :math:`u`, where the two
    are linked by some model physics. It consists mainly of a description of
    the physics model, the model-data misfit and regularization functionals,
    the initial state and parameters, and any auxiliary data such as extra
    fields or boundary conditions.

    This object merely specifies what problem to solve, not how to solve it.
    That functionality is contained in the class `InverseSolver` and the
    classes that inherit from it, such as `GradientDescentSolver` and
    `GaussNewtonSolver`.

    At present, this class assumes that the model physics are an elliptic
    PDE that arises from an action principle. It is not equipped to deal
    with, say, the mass conservation equation, which is hyperbolic.
    """

    def __init__(self, model, method, objective, regularization,
                 state_name, state, parameter_name, parameter,
                 model_args={}, dirichlet_ids=[]):
        r"""Initialize the inverse problem

        Parameters
        ----------
        model
            The forward model physics
        method
            The method of `model` to solve the forward model physics
        objective
            A python function that returns the model-data misfit functional
        regularization
            A python function that returns the regularization functional,
            i.e. the penalty for unphysical parameter fields
        state_name : str
            The name of the state variable as expected by `model.solve`
        state : firedrake.Function
            The initial value of the state variable
        parameter_name : str
            The name of the parameter variable as expected by `model.solve`
        parameter : firedrake.Function
            The initial value of the parameter variable
        model_args : dict, optional
            Any additional arguments to `model.solve`
        dirichlet_ids : list of int, optional
            IDs of points on the domain boundary where Dirichlet conditions
            are applied

        The state variable must be an argument of the objective functional,
        and the parameter variable must be an argument of the
        regularization functional.
        """
        self.model = model
        self.method = method

        self.model_args = model_args
        self.dirichlet_ids = dirichlet_ids

        self.parameter_name = parameter_name
        self.parameter = parameter
        self.state_name = state_name
        self.state = state

        self.objective = objective
        self.regularization = regularization


class InverseSolver(object):
    r"""Base class for approximating the solution of an inverse problem

    This object stores most of the data needed to iteratively optimize the
    value of some parameter of a model, such as the rheology or friction of a
    glacier, to match remote sensing observations. The key variables that need
    to be stored are the current guess for the parameter, the observable state
    computed from that parameter, the adjoint state, and the search direction.

    The optimization problem is solved using a line search method; at each step
    the parameter :math:`p_k` is updated by finding a value :math:`\alpha_k`
    such that

    .. math::
        p_{k + 1} = p_k + \alpha_k\cdot q_k

    reduces the value of the objective function, where :math:`q_k` is the
    search direction. This object implements most of the procedures necessary
    for keeping the parameter, state, and adjoint consistent. Objects that
    inherit from this one only need to define how the search direction is
    computed.
    """
    def _setup(self, problem, callback=(lambda s: None)):
        self._problem = problem
        self._callback = callback

        self._p = problem.parameter.copy(deepcopy=True)
        self._u = problem.state.copy(deepcopy=True)

        self._model_args = dict(**problem.model_args,
                                dirichlet_ids=problem.dirichlet_ids)
        u_name, p_name = problem.state_name, problem.parameter_name
        args = dict(**self._model_args, **{u_name: self._u, p_name: self._p})

        # Make the form compiler use a reasonable number of quadrature points
        degree = problem.model.quadrature_degree(**args)
        self._fc_params = {'quadrature_degree': degree}

        # Create the error, regularization, and barrier functionals
        self._E = problem.objective(self._u)
        self._R = problem.regularization(self._p)
        self._J = self._E + self._R

        # Create the weak form of the forward model, the adjoint state, and
        # the derivative of the objective functional
        self._F = derivative(problem.model.action(**args), self._u)
        self._dF_du = derivative(self._F, self._u)

        # Create a search direction
        dR = derivative(self._R, self._p)
        self._solver_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        Q = self._p.function_space()
        self._q = firedrake.Function(Q)

        # Create the adjoint state variable
        V = self.state.function_space()
        self._λ = firedrake.Function(V)
        dF_dp = derivative(self._F, self._p)

        # Create Dirichlet BCs where they apply for the adjoint solve
        rank = self._λ.ufl_element().num_sub_elements()
        if rank == 0:
            zero = firedrake.Constant(0)
        else:
            zero = firedrake.as_vector((0,) * rank)
        self._bc = firedrake.DirichletBC(V, zero, problem.dirichlet_ids)

        # Create the derivative of the objective functional
        self._dE = derivative(self._E, self._u)
        dR = derivative(self._R, self._p)
        self._dJ = (action(adjoint(dF_dp), self._λ) + dR)

    @property
    def problem(self):
        r"""The instance of the inverse problem we're solving"""
        return self._problem

    @property
    def parameter(self):
        r"""The current value of the parameter we're estimating"""
        return self._p

    @property
    def state(self):
        r"""The state variable computed from the current value of the
        parameter"""
        return self._u

    @property
    def adjoint_state(self):
        r"""The adjoint state variable computed from the current value of
        the parameters and the primal state"""
        return self._λ

    @property
    def search_direction(self):
        r"""Return the direction along which we'll search for a new value of
        the parameters"""
        return self._q

    @property
    def objective(self):
        r"""The functional of the state variable that we're minimizing"""
        return self._E

    @property
    def regularization(self):
        r"""The regularization functional, which penalizes unphysical modes
        in the inferred parameter"""
        return self._R

    @property
    def gradient(self):
        r"""The derivative of the Lagrangian (objective + regularization +
        physics constraints) with respect to the parameter"""
        return self._dJ

    def _forward_solve(self, p):
        method = self.problem.method
        model = self.problem.model
        args = self._model_args
        return method(model, **args, **{self.problem.parameter_name: p})

    def _assemble(self, *args, **kwargs):
        return firedrake.assemble(*args, **kwargs,
                                  form_compiler_parameters=self._fc_params)

    def update_state(self):
        r"""Update the observable state for a new value of the parameters"""
        u, p = self.state, self.parameter
        u.assign(self._forward_solve(p))

    def update_adjoint_state(self):
        r"""Update the adjoint state for new values of the observable state and
        parameters so that we can calculate derivatives"""
        λ = self.adjoint_state
        L = adjoint(self._dF_du)
        firedrake.solve(L == -self._dE, λ, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

    def line_search(self):
        r"""Perform a line search along the descent direction to get a new
        value of the parameter"""
        u, p, q = self.state, self.parameter, self.search_direction

        s = firedrake.Constant(0)
        p_s = p + s * q
        u_s = u.copy(deepcopy=True)

        def f(t):
            s.assign(t)
            u_s.assign(self._forward_solve(p_s))
            return self._assemble(replace(self._J, {u: u_s, p: p_s}))

        try:
            line_search_options = self._line_search_options
        except AttributeError:
            line_search_options = {}

        bracket = scipy.optimize.bracket(f, xa=0.0, xb=_bracket(f))[:3]
        result = scipy.optimize.minimize_scalar(f, bracket=bracket,
                                                options=line_search_options)

        if not result.success:
            raise ValueError("Line search failed: {}".format(result.message))

        return result.x

    def step(self):
        r"""Perform a line search along the current descent direction to get
        a new value of the parameters, then compute the new state, adjoint,
        and descent direction."""
        p, q = self.parameter, self.search_direction
        t = self.line_search()
        p += t * q
        self.update_state()
        self.update_adjoint_state()
        self.update_search_direction()
        self._callback(self)

    def solve(self, atol=0.0, rtol=1e-6, max_iterations=None):
        r"""Search for a new value of the parameters, stopping once either
        the objective functional gets below a threshold value or stops
        improving."""
        max_iterations = max_iterations or np.inf
        J_initial = np.inf

        for iteration in range(max_iterations):
            J = self._assemble(self._J)
            if ((J_initial - J) < rtol * J_initial) or (J <= atol):
                return iteration
            J_initial = J

            self.step()

        return max_iterations


class GradientDescentSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using the objective function gradient
    directly for a search direction

    This implementation of inverse solvers uses the search direction

    .. math::
        q = -M^{-1}dJ

    where :math:`M` is the finite element mass matrix and :math:`dJ` is the
    gradient of the objective functional. The search direction is easy to
    compute using this method, but is often poorly scaled, resulting in more
    expensive bracketing and line search phases."""
    def __init__(self, problem, callback=(lambda s: None)):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()
        self.update_search_direction()
        self._callback(self)

    def update_search_direction(self):
        r"""Set the search direction to be the inverse of the mass matrix times
        the gradient of the objective"""
        q, dJ = self.search_direction, self.gradient
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        firedrake.solve(M == -dJ, q,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)


class GaussNewtonSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using an approximation to the Hessian
    of the objective functional to approach Newton-like efficiency

    This implementation of inverse solvers uses the search direction

    .. math::
        q = -H^{-1}dJ

    where :math:`H` is the Gauss-Newton approximation to the Hessian of the
    objective functional. If :math:`E` is the model-data misfit, :math:`R` is
    the regularization, and :math:`G` is the linearization of the parameter-to-
    observation map, then the Gauss-Newton matrix is

    .. math::
        H = dG^*\cdot d^2E\cdot dG + d^2R.

    This matrix consists of only those terms in the Hessian of the full
    objective functional that are of "first order", i.e. any terms involving
    :math:`d^2G` are dropped. This search direction is more expensive to solve
    for than in, say, gradient descent. However, it is almost always properly
    scaled to the dimensions of the problem and converges in far fewer
    iterations.
    """
    def __init__(self, problem, callback=(lambda s: None),
                 search_tolerance=1e-3):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        self._search_tolerance = search_tolerance
        self._line_search_options = {'xtol': search_tolerance / 2}
        self.update_search_direction()

        self._callback(self)

    def gauss_newton_mult(self, q):
        """Multiply a field by the Gauss-Newton operator"""
        u, p = self.state, self.parameter

        dE = derivative(self._E, u)
        dR = derivative(self._R, p)
        dF_du, dF_dp = self._dF_du, derivative(self._F, p)

        w = firedrake.Function(u.function_space())
        firedrake.solve(dF_du == action(dF_dp, q), w, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        v = firedrake.Function(u.function_space())
        firedrake.solve(adjoint(dF_du) == derivative(dE, u, w), v, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        return action(adjoint(dF_dp), v) + derivative(dR, p, q)

    def gauss_newton_energy_norm(self, q):
        r"""Compute the energy norm of a field w.r.t. the Gauss-Newton operator

        The energy norm of a field :math:`q` w.r.t. the Gauss-Newton operator
        :math:`H` can be computed using one fewer linear solve than if we were
        to calculate the action of :math:`H\cdot q` on :math:`q`. This saves
        computation when using the conjugate gradient method to solve for the
        search direction.
        """
        u, p = self.state, self.parameter

        dE = derivative(self._E, u)
        dR = derivative(self._R, p)
        dF_du, dF_dp = self._dF_du, derivative(self._F, p)

        v = firedrake.Function(u.function_space())
        firedrake.solve(dF_du == action(dF_dp, q), v, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        return self._assemble(firedrake.energy_norm(derivative(dE, u), v) +
                              firedrake.energy_norm(derivative(dR, p), q))

    def update_search_direction(self):
        r"""Solve the Gauss-Newton system for the new search direction using
        the preconditioned conjugate gradient method"""
        p, q, dJ = self.parameter, self.search_direction, self.gradient

        dR = derivative(self.regularization, self.parameter)
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx + \
            derivative(dR, p)

        # Compute the preconditioned residual
        z = firedrake.Function(Q)
        firedrake.solve(M == -dJ, z,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        # This variable is a search direction for a search direction, which
        # is definitely not confusing at all.
        s = z.copy(deepcopy=True)
        q *= 0.0

        old_cost = np.inf
        while True:
            z_mnorm = self._assemble(firedrake.energy_norm(M, z))
            s_hnorm = self.gauss_newton_energy_norm(s)
            α = z_mnorm / s_hnorm

            δz = firedrake.Function(Q)
            g = self.gauss_newton_mult(s)
            firedrake.solve(M == g, δz,
                            solver_parameters=self._solver_params,
                            form_compiler_parameters=self._fc_params)

            q += α * s
            z -= α * δz

            β = self._assemble(firedrake.energy_norm(M, z)) / z_mnorm
            s *= β
            s += z

            energy_norm = self.gauss_newton_energy_norm(q)
            cost = 0.5 * energy_norm + self._assemble(action(dJ, q))

            if (abs(old_cost - cost) / (0.5 * energy_norm)
                    < self._search_tolerance):
                return

            old_cost = cost


class BFGSSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using the limited-memory BFGS method
    to compute a search direction

    This implementation of inverse solvers uses a search direction based on the
    last `m` values of the parameter and objective gradient to construct a low-
    rank approximation to the inverse of the Hessian of the objective. The
    resulting iteration exhibits superlinear convergence, while the search
    direction is only marginally more expensive to compute than the steepest
    descent direction.

    See chapters 6-7 of Nocedal and Wright, Numerical Optimization, 2nd ed.
    """
    def __init__(self, problem, callback=(lambda s: None), memory=5):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        Q = self.parameter.function_space()
        self._memory = memory

        q, dJ = self.search_direction, self.gradient
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        firedrake.solve(M == -dJ, q,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        self._rho = []
        self._ps = [self.parameter.copy(deepcopy=True)]
        self._fs = [q.copy(deepcopy=True)]
        self._fs[-1] *= -1

        self._callback(self)

    @property
    def memory(self):
        r"""Return the number of previous iterations used to construct the low-
        rank approximation to the Hessian"""
        return self._memory

    def update_search_direction(self):
        r"""Apply the low-rank approximation of the Hessian inverse

        This procedure implements the two-loop recursion algorithm to apply the
        low-rank approximation of the Hessian inverse to the derivative of the
        objective functional. See Nocedal and Wright, Numerical Optimization,
        2nd ed., algorithm 7.4."""
        p, q, dJ = self.parameter, self.search_direction, self.gradient
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        f = firedrake.Function(Q)
        firedrake.solve(M == dJ, f,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        # Append the latest values of the parameters and the objective gradient
        # and compute the curvature factor
        ps, fs, ρ = self._ps, self._fs, self._rho
        ρ.append(1 / self._assemble((p - ps[-1]) * (f - fs[-1]) * dx))
        ps.append(p.copy(deepcopy=True))
        fs.append(f.copy(deepcopy=True))

        # Forget any old values of the parameters and objective gradient
        ps = ps[-(self.memory + 1):]
        fs = fs[-(self.memory + 1):]
        ρ = ρ[-self.memory:]

        g = f.copy(deepcopy=True)
        m = len(ρ)
        α = np.zeros(m)
        for i in range(m - 1, -1, -1):
            α[i] = ρ[i] * self._assemble(f * (ps[i + 1] - ps[i]) * dx)
            g -= α[i] * (fs[i + 1] - fs[i])

        r = g.copy(deepcopy=True)
        dp, df = ps[-1] - ps[-2], fs[-1] - fs[-2]
        r *= self._assemble(dp * df * dx) / self._assemble(df * df * dx)

        for i in range(m):
            β = ρ[i] * self._assemble((fs[i + 1] - fs[i]) * r * dx)
            r += (α[i] - β) * (ps[i + 1] - ps[i])

        q.assign(-r)
