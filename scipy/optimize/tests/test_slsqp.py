"""
Unit test for SLSQP optimization.
"""
from numpy.testing import (assert_, assert_array_almost_equal,
                           assert_allclose, assert_equal)
from pytest import raises as assert_raises
import pytest
import numpy as np

from scipy.optimize import (fmin_slsqp, minimize, Bounds, NonlinearConstraint,
                            rosen)
from scipy.optimize._numdiff import approx_derivative


class MyCallBack:
    """pass a custom callback function

    This makes sure it's being used.
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x):
        self.been_called = True
        self.ncalls += 1


def _convert_bounds_to_constraints(bounds, n):
    """Convert bounds to constraints defined as functions."""
    if isinstance(bounds, Bounds):
        bounds = np.hstack((bounds.lb.reshape(-1, 1), bounds.ub.reshape(-1, 1)))
    if bounds is None or len(bounds) == 0:
        return []
    if np.size(bounds) not in (2, 2*n):
        raise ValueError('bounds must be an array_like of size 2 or 2*n')

    # Convert list or tuple of bounds to arrays
    bounds = np.array(bounds, dtype=float)
    bounds = np.resize(bounds, (n, 2))
    finite_lb = np.isfinite(bounds[:, 0])
    finite_ub = np.isfinite(bounds[:, 1])

    # Functions always return one (which is a satisfied inequality) for infinite
    # bounds. Returning np.nans or np.infs doesn't work well with _check_kkt.
    def lb(x):
        g = np.ones(n)
        g[finite_lb] = x[finite_lb] - bounds[finite_lb, 0]
        return g

    def ub(x):
        g = np.ones(n)
        g[finite_ub] = bounds[finite_ub, 1] - x[finite_ub]
        return g

    return [lb, ub]

def _check_kkt(res, constraints=[], bounds=[], atol=1e-06):
    """Checks KKT conditions. See
    https://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions#Matrix_representation
    """
    x = res.x
    gradf = res.jac
    mus = res.kkt['ineq']
    lams = res.kkt['eq']

    if isinstance(constraints, dict):
        constraints = [constraints]

    gs = [constraint['fun'] for constraint in constraints
          if constraint['type'] == 'ineq']
    hs = [constraint['fun'] for constraint in constraints
          if constraint['type'] == 'eq']

    bound_funs = _convert_bounds_to_constraints(bounds, x.shape[0])
    if len(bound_funs):
        gs = gs + bound_funs
        mus = mus + [res.kkt['bounds']['lb'], res.kkt['bounds']['ub']]

    DgTmu = []
    for mu, g in zip(mus, gs):
        g_eval = np.atleast_1d(g(x))
        # primal feasibility, with a small tolerance for floating point error
        np.testing.assert_array_less(-1e-12, g_eval)
        # dual feasibility, with a small tolerance for floating point error
        np.testing.assert_array_less(-1e-12, mu)
        # complementary slackness
        assert_allclose(g_eval @ mu, 0, atol=atol)

        Dg = np.atleast_2d(approx_derivative(g, x, f0=g_eval))
        DgTmu.append(Dg.T @ mu)

    DhTlam = []
    for lam, h in zip(lams, hs):
        # primal feasibility
        assert_allclose(h(x), 0, atol=atol)

        Dh = np.atleast_2d(approx_derivative(h, x))
        DhTlam.append(Dh.T @ lam)

    # stationarity
    assert_allclose(gradf - np.sum(DgTmu, axis=0) - np.sum(DhTlam, axis=0),
                    0, atol=atol)


class TestSLSQP:
    """
    Test SLSQP algorithm using Example 14.4 from Numerical Methods for
    Engineers by Steven Chapra and Raymond Canale.
    This example maximizes the function f(x) = 2*x*y + 2*x - x**2 - 2*y**2,
    which has a maximum at x=2, y=1.
    """
    def setup_method(self):
        self.opts = {'disp': False}

    def fun(self, d, sign=1.0):
        """
        Arguments:
        d     - A list of two elements, where d[0] represents x and d[1] represents y
                 in the following equation.
        sign - A multiplier for f. Since we want to optimize it, and the SciPy
               optimizers can only minimize functions, we need to multiply it by
               -1 to achieve the desired solution
        Returns:
        2*x*y + 2*x - x**2 - 2*y**2

        """
        x = d[0]
        y = d[1]
        return sign*(2*x*y + 2*x - x**2 - 2*y**2)

    def jac(self, d, sign=1.0):
        """
        This is the derivative of fun, returning a NumPy array
        representing df/dx and df/dy.

        """
        x = d[0]
        y = d[1]
        dfdx = sign*(-2*x + 2*y + 2)
        dfdy = sign*(2*x - 4*y)
        return np.array([dfdx, dfdy], float)

    def fun_and_jac(self, d, sign=1.0):
        return self.fun(d, sign), self.jac(d, sign)

    def f_eqcon(self, x, sign=1.0):
        """ Equality constraint """
        return np.array([x[0] - x[1]])

    def fprime_eqcon(self, x, sign=1.0):
        """ Equality constraint, derivative """
        return np.array([[1, -1]])

    def f_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint """
        return self.f_eqcon(x, sign)[0]

    def fprime_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint, derivative """
        return self.fprime_eqcon(x, sign)[0].tolist()

    def f_ieqcon(self, x, sign=1.0):
        """ Inequality constraint """
        return np.array([x[0] - x[1] - 1.0])

    def fprime_ieqcon(self, x, sign=1.0):
        """ Inequality constraint, derivative """
        return np.array([[1, -1]])

    def f_ieqcon2(self, x):
        """ Vector inequality constraint """
        return np.asarray(x)

    def fprime_ieqcon2(self, x):
        """ Vector inequality constraint, derivative """
        return np.identity(x.shape[0])

    # minimize
    def test_minimize_unbounded_approximated(self):
        # Minimize, method='SLSQP': unbounded, approximated jacobian.
        jacs = [None, False, '2-point', '3-point']
        for jac in jacs:
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac, method='SLSQP',
                           options=self.opts)
            assert_(res['success'], res['message'])
            assert_allclose(res.x, [2, 1])

    def test_minimize_unbounded_given(self):
        # Minimize, method='SLSQP': unbounded, given Jacobian.
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       jac=self.jac, method='SLSQP', options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_minimize_bounded_approximated(self):
        # Minimize, method='SLSQP': bounded, approximated jacobian.
        jacs = [None, False, '2-point', '3-point']
        bnds = ((2.5, None), (None, 0.5))
        for jac in jacs:
            with np.errstate(invalid='ignore'):
                res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                               jac=jac,
                               bounds=bnds,
                               method='SLSQP', options=self.opts)
            assert_(res['success'], res['message'])
            assert_allclose(res.x, [2.5, 0.5])
            _check_kkt(res, bounds=bnds)

    def test_minimize_unbounded_combined(self):
        # Minimize, method='SLSQP': unbounded, combined function and Jacobian.
        res = minimize(self.fun_and_jac, [-1.0, 1.0], args=(-1.0, ),
                       jac=True, method='SLSQP', options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_minimize_equality_approximated(self):
        # Minimize with method='SLSQP': equality constraint, approx. jacobian.
        jacs = [None, False, '2-point', '3-point']
        cons = {'type': 'eq', 'fun': self.f_eqcon, 'args': (-1.0,)}
        for jac in jacs:
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac,
                           constraints=cons,
                           method='SLSQP', options=self.opts)
            assert_(res['success'], res['message'])
            assert_allclose(res.x, [1, 1])
            _check_kkt(res, constraints=cons)

    def test_minimize_equality_given(self):
        # Minimize with method='SLSQP': equality constraint, given Jacobian.
        cons = {'type': 'eq', 'fun':self.f_eqcon, 'args': (-1.0,)}
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method='SLSQP', args=(-1.0,),
                       constraints=cons,
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])
        _check_kkt(res, constraints=cons)

    def test_minimize_equality_given2(self):
        # Minimize with method='SLSQP': equality constraint, given Jacobian
        # for fun and const.
        cons = {'type': 'eq', 'fun': self.f_eqcon, 'args': (-1.0,),
                'jac': self.fprime_eqcon}
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0,),
                       constraints=cons,
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])
        _check_kkt(res, constraints=cons)

    def test_minimize_equality_given_cons_scalar(self):
        # Minimize with method='SLSQP': scalar equality constraint, given
        # Jacobian for fun and const.
        cons = {'type': 'eq', 'fun': self.f_eqcon_scalar, 'args': (-1.0,),
                'jac': self.fprime_eqcon_scalar}
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0,),
                       constraints=cons,
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])
        _check_kkt(res, constraints=cons)

    def test_minimize_inequality_given(self):
        # Minimize with method='SLSQP': inequality constraint, given Jacobian.
        cons = {'type': 'ineq', 'fun': self.f_ieqcon, 'args': (-1.0,)}
        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0, ),
                       constraints=cons,
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1], atol=1e-3)
        _check_kkt(res, constraints=cons, atol=1e-03)

    def test_minimize_inequality_given_vector_constraints(self):
        # Minimize with method='SLSQP': vector inequality constraint, given
        # Jacobian.
        cons = {'type': 'ineq', 'fun': self.f_ieqcon2,
                'jac': self.fprime_ieqcon2}
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method='SLSQP', args=(-1.0,),
                       constraints=cons,
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])
        _check_kkt(res, constraints=cons)

    def test_minimize_bounded_constraint(self):
        # when the constraint makes the solver go up against a parameter
        # bound make sure that the numerical differentiation of the
        # jacobian doesn't try to exceed that bound using a finite difference.
        # gh11403
        def c(x):
            assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
            return x[0] ** 0.5 + x[1]

        def f(x):
            assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
            return -x[0] ** 2 + x[1] ** 2

        cns = [NonlinearConstraint(c, 0, 1.5)]
        x0 = np.asarray([0.9, 0.5])
        bnd = Bounds([0., 0.], [1.0, 1.0])
        minimize(f, x0, method='SLSQP', bounds=bnd, constraints=cns)

    def test_minimize_bound_equality_given2(self):
        # Minimize with method='SLSQP': bounds, eq. const., given jac. for
        # fun. and const.
        bnds = [(-0.8, 1.), (-1, 0.8)]
        cons = {'type': 'eq', 'fun': self.f_eqcon, 'args': (-1.0,),
                'jac': self.fprime_eqcon}
        bnd_cons = _convert_bounds_to_constraints(bnds, 2)
        cons = [cons, {'type': 'ineq', 'fun': bnd_cons[0]}, {'type': 'ineq', 'fun': bnd_cons[1]}]

        res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
                       jac=self.jac, args=(-1.0,),
                       bounds=bnds,
                       constraints=cons,
                       options=self.opts)

        assert_(res['success'], res['message'])
        assert_allclose(res.x, [0.8, 0.8], atol=1e-3)
        _check_kkt(res, cons, bnds)

    # fmin_slsqp
    def test_unbounded_approximated(self):
        # SLSQP: unbounded, approximated Jacobian.
        res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [2, 1])

    def test_unbounded_given(self):
        # SLSQP: unbounded, given Jacobian.
        res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
                         fprime = self.jac, iprint = 0,
                         full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [2, 1])

    def test_equality_approximated(self):
        # SLSQP: equality constraint, approximated Jacobian.
        res = fmin_slsqp(self.fun,[-1.0,1.0], args=(-1.0,),
                         eqcons = [self.f_eqcon],
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [1, 1])

    def test_equality_given(self):
        # SLSQP: equality constraint, given Jacobian.
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0,),
                         eqcons = [self.f_eqcon], iprint = 0,
                         full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [1, 1])

    def test_equality_given2(self):
        # SLSQP: equality constraint, given Jacobian for fun and const.
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0,),
                         f_eqcons = self.f_eqcon,
                         fprime_eqcons = self.fprime_eqcon,
                         iprint = 0,
                         full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [1, 1])

    def test_inequality_given(self):
        # SLSQP: inequality constraint, given Jacobian.
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0, ),
                         ieqcons = [self.f_ieqcon],
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [2, 1], decimal=3)

    def test_bound_equality_given2(self):
        # SLSQP: bounds, eq. const., given jac. for fun. and const.
        res = fmin_slsqp(self.fun, [-1.0, 1.0],
                         fprime=self.jac, args=(-1.0, ),
                         bounds = [(-0.8, 1.), (-1, 0.8)],
                         f_eqcons = self.f_eqcon,
                         fprime_eqcons = self.fprime_eqcon,
                         iprint = 0, full_output = 1)
        x, fx, its, imode, smode = res
        assert_(imode == 0, imode)
        assert_array_almost_equal(x, [0.8, 0.8], decimal=3)
        assert_(-0.8 <= x[0] <= 1)
        assert_(-1 <= x[1] <= 0.8)

    def test_scalar_constraints(self):
        # Regression test for gh-2182
        x = fmin_slsqp(lambda z: z**2, [3.],
                       ieqcons=[lambda z: z[0] - 1],
                       iprint=0)
        assert_array_almost_equal(x, [1.])

        x = fmin_slsqp(lambda z: z**2, [3.],
                       f_ieqcons=lambda z: [z[0] - 1],
                       iprint=0)
        assert_array_almost_equal(x, [1.])

    def test_integer_bounds(self):
        # This should not raise an exception
        fmin_slsqp(lambda z: z**2 - 1, [0], bounds=[[0, 1]], iprint=0)

    def test_array_bounds(self):
        # NumPy used to treat n-dimensional 1-element arrays as scalars
        # in some cases.  The handling of `bounds` by `fmin_slsqp` still
        # supports this behavior.
        bounds = [(-np.inf, np.inf), (np.array([2]), np.array([3]))]
        x = fmin_slsqp(lambda z: np.sum(z**2 - 1), [2.5, 2.5], bounds=bounds,
                       iprint=0)
        assert_array_almost_equal(x, [0, 2])

    def test_obj_must_return_scalar(self):
        # Regression test for Github Issue #5433
        # If objective function does not return a scalar, raises ValueError
        with assert_raises(ValueError):
            fmin_slsqp(lambda x: [0, 1], [1, 2, 3])

    def test_obj_returns_scalar_in_list(self):
        # Test for Github Issue #5433 and PR #6691
        # Objective function should be able to return length-1 Python list
        #  containing the scalar
        fmin_slsqp(lambda x: [0], [1, 2, 3], iprint=0)

    def test_callback(self):
        # Minimize, method='SLSQP': unbounded, approximated jacobian. Check for callback
        callback = MyCallBack()
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       method='SLSQP', callback=callback, options=self.opts)
        assert_(res['success'], res['message'])
        assert_(callback.been_called)
        assert_equal(callback.ncalls, res['nit'])

    def test_inconsistent_linearization(self):
        # SLSQP must be able to solve this problem, even if the
        # linearized problem at the starting point is infeasible.

        # Linearized constraints are
        #
        #    2*x0[0]*x[0] >= 1
        #
        # At x0 = [0, 1], the second constraint is clearly infeasible.
        # This triggers a call with n2==1 in the LSQ subroutine.
        x = [0, 1]
        def f1(x):
            return x[0] + x[1] - 2
        def f2(x):
            return x[0] ** 2 - 1

        cons = ({'type': 'eq', 'fun': f1}, {'type': 'ineq', 'fun': f2})
        bnds = ((0, None), (0, None))

        sol = minimize(lambda x: x[0]**2 + x[1]**2,
                       x, constraints=cons, bounds=bnds, method='SLSQP')
        x = sol.x

        assert_allclose(f1(x), 0, atol=1e-8)
        assert_(f2(x) >= -1e-8)
        assert_(sol.success, sol)
        _check_kkt(sol, cons, bnds)

    def test_regression_5743(self):
        # SLSQP must not indicate success for this problem,
        # which is infeasible.
        x = [1, 2]
        sol = minimize(
            lambda x: x[0]**2 + x[1]**2,
            x,
            constraints=({'type':'eq','fun': lambda x: x[0]+x[1]-1},
                         {'type':'ineq','fun': lambda x: x[0]-2}),
            bounds=((0,None), (0,None)),
            method='SLSQP')
        assert_(not sol.success, sol)

    def test_gh_6676(self):
        def func(x):
            return (x[0] - 1)**2 + 2*(x[1] - 1)**2 + 0.5*(x[2] - 1)**2

        sol = minimize(func, [0, 0, 0], method='SLSQP')
        assert_(sol.jac.shape == (3,))

    def test_invalid_bounds(self):
        # Raise correct error when lower bound is greater than upper bound.
        # See Github issue 6875.
        bounds_list = [
            ((1, 2), (2, 1)),
            ((2, 1), (1, 2)),
            ((2, 1), (2, 1)),
            ((np.inf, 0), (np.inf, 0)),
            ((1, -np.inf), (0, 1)),
        ]
        for bounds in bounds_list:
            with assert_raises(ValueError):
                minimize(self.fun, [-1.0, 1.0], bounds=bounds, method='SLSQP')

    def test_bounds_clipping(self):
        #
        # SLSQP returns bogus results for initial guess out of bounds, gh-6859
        #
        def f(x):
            return (x[0] - 1)**2

        bnds = [(None, 0)]
        sol = minimize(f, [10], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

        sol = minimize(f, [-10], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

        bnds = [(2, None)]
        sol = minimize(f, [-10], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

        sol = minimize(f, [10], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

        bnds = [(-1, 0)]
        sol = minimize(f, [-0.5], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

        sol = minimize(f, [10], method='slsqp', bounds=bnds)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, bounds=bnds)

    def test_infeasible_initial(self):
        # Check SLSQP behavior with infeasible initial point
        def f(x):
            x, = x
            return x*x - 2*x + 1

        cons_u = [{'type': 'ineq', 'fun': lambda x: 0 - x}]
        cons_l = [{'type': 'ineq', 'fun': lambda x: x - 2}]
        cons_ul = [{'type': 'ineq', 'fun': lambda x: 0 - x},
                   {'type': 'ineq', 'fun': lambda x: x + 1}]

        sol = minimize(f, [10], method='slsqp', constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, constraints=cons_u)

        sol = minimize(f, [-10], method='slsqp', constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)
        _check_kkt(sol, constraints=cons_l)

        sol = minimize(f, [-10], method='slsqp', constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, constraints=cons_u)

        sol = minimize(f, [10], method='slsqp', constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=1e-10)
        _check_kkt(sol, constraints=cons_l)

        sol = minimize(f, [-0.5], method='slsqp', constraints=cons_ul)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, constraints=cons_ul)

        sol = minimize(f, [10], method='slsqp', constraints=cons_ul)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=1e-10)
        _check_kkt(sol, constraints=cons_ul)

    def test_inconsistent_inequalities(self):
        # gh-7618

        def cost(x):
            return -1 * x[0] + 4 * x[1]

        def ineqcons1(x):
            return x[1] - x[0] - 1

        def ineqcons2(x):
            return x[0] - x[1]

        # The inequalities are inconsistent, so no solution can exist:
        #
        # x1 >= x0 + 1
        # x0 >= x1

        x0 = (1,5)
        bounds = ((-5, 5), (-5, 5))
        cons = (dict(type='ineq', fun=ineqcons1), dict(type='ineq', fun=ineqcons2))
        res = minimize(cost, x0, method='SLSQP', bounds=bounds, constraints=cons)

        assert_(not res.success)

    def test_new_bounds_type(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2
        bounds = Bounds([1, 0], [np.inf, np.inf])
        sol = minimize(f, [0, 0], method='slsqp', bounds=bounds)
        assert_(sol.success)
        assert_allclose(sol.x, [1, 0])
        _check_kkt(sol, bounds=bounds)

    def test_nested_minimization(self):

        class NestedProblem():

            def __init__(self):
                self.F_outer_count = 0

            def F_outer(self, x):
                self.F_outer_count += 1
                if self.F_outer_count > 1000:
                    raise Exception("Nested minimization failed to terminate.")
                inner_res = minimize(self.F_inner, (3, 4), method="SLSQP")
                assert_(inner_res.success)
                assert_allclose(inner_res.x, [1, 1])
                return x[0]**2 + x[1]**2 + x[2]**2

            def F_inner(self, x):
                return (x[0] - 1)**2 + (x[1] - 1)**2

            def solve(self):
                outer_res = minimize(self.F_outer, (5, 5, 5), method="SLSQP")
                assert_(outer_res.success)
                assert_allclose(outer_res.x, [0, 0, 0])

        problem = NestedProblem()
        problem.solve()

    def test_gh1758(self):
        # the test suggested in gh1758
        # https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/
        # implement two equality constraints, in R^2.
        def fun(x):
            return np.sqrt(x[1])

        def f_eqcon(x):
            """ Equality constraint """
            return x[1] - (2 * x[0]) ** 3

        def f_eqcon2(x):
            """ Equality constraint """
            return x[1] - (-x[0] + 1) ** 3

        c1 = {'type': 'eq', 'fun': f_eqcon}
        c2 = {'type': 'eq', 'fun': f_eqcon2}
        constraints = [c1, c2]
        bounds = [(-0.5, 1), (0, 8)]

        res = minimize(fun, [8, 0.25], method='SLSQP', constraints=constraints,
                       bounds=bounds)

        assert_allclose(res.fun, 0.5443310539518)
        assert_allclose(res.x, [0.33333333, 0.2962963])
        _check_kkt(res, constraints, bounds)
        assert res.success

    def test_gh9640(self):
        np.random.seed(10)
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - 3},
                {'type': 'ineq', 'fun': lambda x: x[1] + x[2] - 2})
        bnds = ((-2, 2), (-2, 2), (-2, 2))

        def target(x):
            return 1
        x0 = [-1.8869783504471584, -0.640096352696244, -0.8174212253407696]
        res = minimize(target, x0, method='SLSQP', bounds=bnds,
                       constraints=cons, options={'maxiter': 10000})

        # The problem is infeasible, so it cannot succeed
        assert not res.success

    def test_parameters_stay_within_bounds(self):
        # gh11403. For some problems the SLSQP Fortran code suggests a step
        # outside one of the lower/upper bounds. When this happens
        # approx_derivative complains because it's being asked to evaluate
        # a gradient outside its domain.
        np.random.seed(1)
        bounds = Bounds(np.array([0.1]), np.array([1.0]))
        n_inputs = len(bounds.lb)
        x0 = np.array(bounds.lb + (bounds.ub - bounds.lb) *
                      np.random.random(n_inputs))

        def f(x):
            assert (x >= bounds.lb).all()
            return np.linalg.norm(x)

        with pytest.warns(RuntimeWarning, match='x were outside bounds'):
            res = minimize(f, x0, method='SLSQP', bounds=bounds)
            assert res.success
            _check_kkt(res, bounds=bounds)

    def test_kkt_equality(self):
        # An equality constraint mentioned in gh9839
        # Add a bound on a second, independent variable to test indexing
        def fun(x):
            return np.sum(x ** 2)

        def con_fun(x):
            return x[0] - 1

        cons = [{'fun': con_fun, 'type': 'eq'}]
        bnds = [(None, None), (1.0, None)]
        x0 = [3.0, 4.0]
        res = minimize(fun, x0, method='SLSQP', constraints=cons, bounds=bnds)

        assert_allclose(res.kkt['eq'][0], np.array([2.0]))
        _check_kkt(res, cons, bnds)

    def test_kkt_inequality(self):
        """Test kkt multiplier return with example from GH14394. These are
        linear inequality constraints that can be specified either as
        constraints or bounds, so we can test both. To test if dimensions
        indices of bounds are correctly extracted, adds a second independent
        variable to the example from GH14394."""

        # The main constraint to be tested
        def con_fun1(x):
            return np.array([x[0] - 1.0, 2.0 - x[0]])

        # Lower bound on x[1]
        def con_fun2(x):
            return x[1:] - 0.75

        cons = [{'type': 'ineq', 'fun': con_fun1},
                {'type': 'ineq', 'fun': con_fun2}]

        # x[0] - 1 >=0 and 2 - x[0] >= 0. Equivalently, 1 <= x[0] <= 2.
        bnds = [(1.0, 2.0), (0.75, None)]

        x0 = np.array([1.5, 1.5])

        # Test cases for c < 1, 1 < c < 2, and c > 2
        for c in [0.6, 1.5, 2.3]:
            def fun(x):
                return (x[:1] - c) ** 2 + x[1] ** 2

            def jac(x):
                return np.array([2 * (x[0] - c), 2 * x[1]])

            # Test with constraints specified using constraint keyword
            res_cons = minimize(fun, x0, method='SLSQP', jac=jac,
                                constraints=cons)
            w_cons = res_cons.kkt['ineq']
            _check_kkt(res_cons, constraints=cons)

            # Test with same constraints specified using bounds keyword
            res_bnds = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds)
            w_bnds = res_bnds.kkt['bounds']
            _check_kkt(res_bnds, bounds=bnds)

            # Verify results are the same in both setups
            assert_allclose(res_cons.x, res_bnds.x, atol=1e-12)
            # Check extra variable bound matches constraint
            assert_allclose(w_cons[1], w_bnds["lb"][1], rtol=1e-06)

            if c < 1:
                analytical = 2 - 2 * c
                assert_allclose(w_cons[0][0], analytical, rtol=1e-06)
                assert_allclose(w_cons[0][0], w_bnds["lb"][0], rtol=1e-06)
            elif c > 2:
                analytical = 2 * c - 4
                assert_allclose(w_cons[0][1], analytical, rtol=1e-06)
                assert_allclose(w_cons[0][1], w_bnds["ub"][0], rtol=1e-06)
            else:
                assert_allclose(w_cons[0], 0., atol=1e-06, rtol=1e-06)
                assert_allclose(w_bnds["lb"][0], 0., atol=1e-06, rtol=1e-06)
                assert_allclose(w_bnds["ub"][0], 0., atol=1e-06, rtol=1e-06)

    def test_kkt_bounds(self):
        # Test that KKT conditions hold with varying numbers of constraints and
        # parameter dimensions. Needed to make sure that indices used for
        # extracting multipliers for bounds are correct.
        def fun(x):
            return np.sum(x ** 2)

        def jac(x):
            return 2. * x

        # Loop over parameter dimensions
        for n in range(1, 5):
            bnds = Bounds(lb=np.ones(n))
            x0 = bnds.lb + 1.0

            # Loop over number of constraints
            for m in range(0, 5):
                if m == 0:
                    cons = []
                else:
                    # Dummy constraints automatically satisfied by bounds
                    def cons_fun(x):
                        g = np.empty(m)
                        for i in range(m):
                            g[i] = i+1 + 2*n - np.sum(x)
                        return g
                    cons = {'type': 'ineq', 'fun': cons_fun}

                res = minimize(fun, x0, method='SLSQP', jac=jac,
                               constraints=cons, bounds=bnds)
                _check_kkt(res, cons, bnds)

    def test_kkt_constrained_rosen(self):

        fun = rosen

        def g(x):
            x0, x1 = x
            c1 = x0 + 2*x1 - 1
            c2 = x0**2 + x1 - 1
            c3 = x0**2 - x1 - 1
            return -np.array([c1, c2, c3])

        def h(x):
            x0, x1 = x
            return np.array([2*x0 + x1 - 1])

        x0 = [0.4149, 0.1701]
        bounds = [(0, 1), (-0.5, 2)]
        constraints = [{'type': 'ineq', 'fun': g},
                       {'type': 'eq', 'fun': h}]
        res = minimize(fun, x0, bounds=bounds, constraints=constraints,
                       method='slsqp')

        _check_kkt(res, constraints, bounds)

    def test_kkt_constrained_stackexchange(self):
        # Check test with example from:
        # https://math.stackexchange.com/questions/3056670/kkt-condition-with-equality-and-inequality-constraints  # noqa

        def fun(x):
            x1, x2 = x
            return (x1 - 3)**2 + (x2 - 2)**2

        def g(x):
            x1, x2 = x
            return -np.array([x1**2 + x2**2 - 5])

        def h(x):
            x1, x2 = x
            return np.array([x1 + 2*x2 - 4])

        x0 = [2, 1]
        bounds = [(0, None), (0, None)]
        constraints = [{'type': 'ineq', 'fun': g},
                       {'type': 'eq', 'fun': h}]
        res = minimize(fun, x0, bounds=bounds, constraints=constraints,
                       method='slsqp')

        _check_kkt(res, constraints, bounds)

    def test_example(self):
        # Verify the example given in the `minimize` documentation
        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

        bnds = ((0, None), (0, None))

        res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
                       constraints = cons)

        kkt0 = res.kkt['ineq'][0]
        f0 = res.fun

        assert_allclose(res.x, [1.4, 1.7])
        assert_allclose(kkt0, 0.8)
        _check_kkt(res, cons, bnds)

        # Tighten the constraints by a small amount eps
        eps = 0.01
        cons[0]['fun'] = lambda x: x[0] - 2 * x[1] + 2 - eps

        res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
                       constraints=cons)

        f1 = res.fun
        assert_allclose(f1 - f0, eps * kkt0, atol=eps**2)
        _check_kkt(res, cons, bnds)
