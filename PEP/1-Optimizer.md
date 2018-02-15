Scientific PEP - Introduction of Optimizer classes
==================================================

* Abstract
* Introduction
    * Here's what minimization does...
        * It minimizes a function
        * These are fairly independent -- functions and optimizers are not tied together.
    * Point to users of...
        * Minimization in general
        * scipy.optimize.minimize (many users, do a github search)
* Problem
    * No standard interface for optimizers or functions.
        * Have to explain why minimize isn't a standard interface.
        * Minimization is a common problem and implemented in many places
        * Providing a standard interface for this could help unify users and libraries
    * Current API needs improvement
       * scipy.optimize.minimize is a black box (have to explain why)
           * hides all details. Some are literal black boxes and implemented in Fortrain/C.
           * would like ability to proceed stepwise through iteration
               * Callback is not sufficient
                   * only sends `x`, not the potentially expensive `f(x), g(x), h(x)`.
                   * What if some internal state is wanted?
               * What if running some web server, and don't have time to wait for minimization to finish?
           * would like to access solver state
               * e.g., current value of f(x)
               * e.g., for coding gradients
          * can't access to solver state or hyper parameters, and change on fly
              * e.g., gradient coding as example
              * e.g, change convergence tolerances as we're going
              * e.g., change mutation constant during differential evolution.
       * addition of new features to minimizers leads to lengthy functions and lots of duplicate code.
           * Classes => inherietance. Base class improves => all improve.
           * Unix philisophy, small sharp tools for one job and one job only. Not many dull tools for the same job.
       * Mixing of function arguments with optimization arguments (plus, there are too many arguments)
       * examine scipy issues database to see what issues would be cleaned up.
           * #5832, grad.T should be returned.
       * no kwargs for func, only args
* Other solutions
    * Class defs: PyTorch, skopt
    * Functional class wrapper around minimize: statsmodels, astropy, scikits.fitting
    * Functional defs: sklearn, daskml
* Solution
    * Classes (briefly list attributes, functions)
    * Goal: provide minimal class for interface.
* Solution enhancements
    * Provide standard interface
        * for enhancements to sklearn, dask-ml, etc. Possibly PyTorch. **Would those projects be prepared to state that?**
        * it would provide a standard way to operate the object, but all the classes would still have different names
    * Provide class features
        * object interaction. Useful for experts, intermediates.
        * expose alg hyperparameters (grid search, etc)
        * keyboard interrupts
   * Clean up minimize API (it's complicated right now)
   * introduction of context manager enables easy setup of cleanup actions
       * would make it easier have wholesale introduction of things like multiprocessing.
* Solution implementation
    * Give an example
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


## Proposed code
``` python
def func(x, *args):
    return x**2 + args[0]

def grad(x, *args):
  return 2 * x

def callback(x):
  print(x)

# existing call has lots of parameters, mixing optimizer args with func args
result = minimize(func, x0, args=(2,), jac=jac, method='BFGS', maxiter=10, callback=callback)

# proposed
function = Function(func=func, args=(2,), kwargs=kwargs, jac=jac)
opt = BFGS(function, x0)
result = opt.solve(maxiter=10, callback=callback)

# could also have
# result = BFGS(function, x0).solve(maxiter=10, callback=callback)

# alternatively control how iteration occurs
d = opt.hyper_parameters
for i, v in enumerate(opt):
  x, f = v
  print(i, f, x)
  d['my_hyper_parameter'] = np.inf

# use function classes encapsulates the whole function and offers the potential for more sophisticated calculation.
class Quad(Function):
   def __init__(self, bkg):
       super(Quad, self).__init__(self)
       self.bkg = bkg

   def func(self, x):
      return x**2 + args[0]
​
   def grad(self, x):
      return 2 * x
​
   def hess(self, x):
      return 2

 opt = BFGS(function, x0).solve(maxiter=10)

 # context managers offer the chance for cleanup actions, for example multiprocessing.
 with DifferentialEvolutionSolver(function, bounds, workers=2) as opt:
     # the __entry__ and __exit__ in the solver can create and close
     # multiprocessing pools.
     res = opt.solve()
   ```

