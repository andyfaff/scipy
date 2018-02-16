Scientific PEP -- Introduction of Optimizer and Function classes
================================================================

.. outline::

   * Abstract
   * Introduction
       * Here's what minimization does...
           * It minimizes a function
           * These are -or should be- fairly independent -- functions and optimizers are not tied together.
       * Point to users of...
           * Minimization in general
           * scipy.optimize.minimize (many users, do a github search)
   * Motivation
       * No standard interface for optimizers or functions.
           * Have to explain why minimize isn't a standard interface.
           * Minimization is a common problem and implemented in many places
           * Providing a standard interface for this could help unify users and libraries
       * Current API needs improvement
          * minimize is trying to be a class
                 * method: should be subclasses
                 * show_options: show method-specific args
                 * some options specific to method (jac, hess, hessp, contraints, options, bounds)
                 * OptimizeResult: trying to expose what should be properties of class
                 * callback: not adequate (only sends one arg, not any internal state)
                      * only sends `x`, not the potentially expensive `f(x), g(x), h(x)`.
                      * What if some internal state is wanted?
          * function arg is trying to be a class
              * jac, hess, hessp
              * args (kwargs?)
          * there is no separation of concerns between function and minimizer
              * meaning the minimizer is carrying out numerical gradient calculations.
              * The correct place for grad computation belongs with the function, not the minimizer.
              * Mixing of function arguments with optimization arguments (plus, there are too many arguments)
              * no kwargs for func, only args
          * scipy.optimize.minimize is a black box (have to explain why)
              * hides all details. Some are literal black boxes and implemented in Fortran/C.
                  * e.g., what if want to change step size? Choosing an initial step size is difficult. There's theoritical
                    bounds, but these are not known in practice.
                  * if the user doesn't provide a gradient function the minimizers currently use the same absolute step size
                      for numerical differentiation for the duration of the minimization. However, the fd-step size should
                      be relative to parameter value as it changes. Not easy to fix this in current implementation without placing
                      the onus on the user to write their own grad function, this is the job of the library.
                      The new Function object will offer more options for numerical differentiation (absolute step, relative
                      step, 2-point/3-point/complex step, bounds). Of course, the user can still provide their own gradient
                      implementation if preferred.
                  * would like ability to proceed stepwise through iteration
                      * What if running some web server, and don't have time to wait for minimization to finish?
                  * would like to access solver state
                      * e.g., current value of f(x)
                      * e.g., for coding gradients
                  * can't access solver state or hyper parameters, and change on fly
                     * e.g. gradient coding as example
                     * e.g. change convergence tolerances as we're going
                     * e.g. change mutation constant during differential evolution.
          * addition of new features to minimizers leads to lengthy functions and lots of duplicate code.
              * Classes => inherietance. Base class improves => all improve.
              * Unix philisophy, small sharp tools for one job and one job only. Not many dull tools for the same job.
          * examine scipy issues database to see what issues would be cleaned up.
              * #5832, grad.T should be returned but not documented
   * Existing work
       * Class defs: PyTorch, skopt
       * Functional class wrapper around minimize: statsmodels, astropy, scikits.fitting
       * Functional defs: sklearn, daskml, skimage
   * Proposed solution
       * Classes (idea: `Function` and `Optimizer` class)
       * Goal:
           * provide minimal class interface
           * preserve backwards compatibility
           * targetted at minimization of scalar functions to start with, although the Optimizer class and its methods should
             be a suitable base class for implementing for class based root and least-squares solvers. For example, both of
             those examples need to iterate, they both finish up with an OptimizeResult, they both have convergence criteria,
             etc.
       * Give an example
   * Enhancements
       * Provide standard interface
           * for enhancements to sklearn, dask-ml, etc. Possibly PyTorch. **Would those projects be prepared to state that?**
           * it would provide a standard way to operate the object, but all the classes would still have different names
           * give example of how sklearn could revamp (ask the developers how they'd use it)
       * Provide class features
           * object interaction. Useful for experts, intermediates.
           * expose alg hyperparameters (grid search, etc)
           * keyboard interrupts
           * introduction of context manager enables easy setup of cleanup actions
              * would make it easier have wholesale introduction of things like multiprocessing.
              * We should think about multiprocessing or multithreaded algorithms like Hogwild!. How will these be used?
      * Clean up minimize API (it's complicated right now)
         * Require fewer arguments to minimize, and separate them
   * Implementation
       * List functions, attributes in more depth
       * Existing code
           * How would it work with C/Fortran optimizers?
           * What interface are we proposing? See proposed code below
       * Speed
         * will be benchmarked to check that performance is not damaged. Class based system is easy to convert to cython.
       * Backwards compatibility
         * backwards compatibility is a focus
         * the functionality will remain but rely on the solver objects. Should be able to remove `_minimize_lbfgsb`, etc.
         * new solver objects can be used by themselves.

*Abstract*

Introduction
============

Optimization is extremely common and often critical in many
applications. Imaging, machine learning, industrial engineering and a
variety of regression problems. As such, it has been adopted by
libraries including SciPy and many related libraries (e.g.,
scikit-learn). Optimization has received significant attention from
industry as well -- Google, Facebook, Amazon and Microsoft have
developed Tensorflow, PyTorch, MXNet and CNTK respectively, all of which
use optimization, have Python bindings and are open source.

Optimization is the minimization or maximization (though typically
minimization) of a certain function. Minimization tries to find which
argument yields the smallest function value, or in pseudo-code,

.. code:: python

    def f(x):
        return (x - 1) ** 2

    x_hat = minimize(f)
    assert x_hat == 1 and f(x_hat) == 0

The SciPy ``minimize`` function has been widely used. Over 17,000
results for "``from scipy.optimize import minimize``" appear from a
GitHub search, and ``minimize`` is included in many libraries including
scikit-learn, scikit-image, statsmodels and astropy. Preserving
backwards compatibility to keep this code functional is a priority.
However, we believe that we can improve upon SciPy's minimization API.
We believe implementation of this will allow easier use, enable more
widespread use and unify various interfaces.

Motivation
==========
No standard interface
---------------------

Current API needs improvement
-----------------------------

`minimize` has many class features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`minimize`'s `func` argument has many class features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`minimize` is a black box
^^^^^^^^^^^^^^^^^^^^^^^^^
Separation of function and minimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Repeated code
^^^^^^^^^^^^^
Open bugs
^^^^^^^^^

Existing work
=============

Proposed solution
=================
Goals
-----
Embodiment
----------
Example
-------

Enhancements
============
Standard interface
------------------
Class features
--------------
API cleaning
------------



Implementation
==============
Definition
----------
Existing code
-------------
Backward compatibility
----------------------

Proposed code
-------------


.. code-block:: python

    def func(x, *args):
        return x**2 + args[0]
    def grad(x, *args):
        return 2 * x

    def callback(x): print(x)

    # existing call has lots of parameters, mixing optimizer args with func args

    result = minimize(func, x0, args=(2,), jac=jac, method='BFGS',
    maxiter=10, callback=callback)

    # proposed

    function = Function(func=func, args=(2,), kwargs=kwargs, jac=jac) opt =
    BFGS(function, x0) result = opt.solve(maxiter=10, callback=callback)

    # could also have

    result = BFGS(function, x0).solve(maxiter=10, callback=callback)

    # alternatively control how iteration occurs

    d = opt.hyper\_parameters for i, v in enumerate(opt): x, f = v print(i,
    f, x) d['my\_hyper\_parameter'] = np.inf

    # use function classes encapsulates the whole function and offers the potential for more sophisticated calculation.

    class Quad(Function):
        def __init__(self, bkg):
            super(Quad, self).__init__(self)
            self.bkg = bkg

        def func(self, x):
            return (x**2 + args[0])

        def grad(self, x):
            return 2*x

        def hess(self, x):
            return 2

    opt = BFGS(function, x0).solve(maxiter=10)

    # context managers offer the chance for cleanup actions, for example
    multiprocessing.

    with DifferentialEvolutionSolver(function, bounds,
    workers=2) as opt:
        # the __entry__ and __exit__ in the solver can create
    and close
        # multiprocessing pools.
        res = opt.solve()
