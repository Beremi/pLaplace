# %%
from pLaplace_1D import test_runner_1D

# all possible types of inputs (not all combinations allowed)
_ = ["numpy", "numba", "jax"]  # "val_grad"
_ = ["numpy_Laplace", "numpy_SFD", "numba_Laplace", "numba_SFD",
     "jax_dense", "jax_sparse", "jax_Laplace", "jax_SFD"]  # "hess"
_ = ["my_newton", "my_trust_region", "scipy_trust-exact", "scipy_Newton-CG_hess",
     "scipy_BFGS", "scipy_CG", "scipy_L-BFGS-B", "scipy_TNC"]  # "minimizer"
_ = ["zero", "laplace"]  # "initial_guess"

problem_setting = {"p": 3,   # p in pLaplace
                   "a": -1,  # left bound of computational domain
                   "b": 1,   # right bound of computational domain
                   "f": "default"}  # source term, constant 10

sizes = [10, 100, 1000]


# %% [markdown]
# # Conclusions first
# - fastest value and grad evaluation have numba (rewrite of matlab code)
# - fastest method without the use of user provided hessian is `L-BFGS-B` (scipy)
# - method similar to the "best" in the matlab code is `Newton-CG` (scipy) with approximation of hessian using numrical
#   differentiation SFD (rewrite of matlab code)
# - fastest method overall is my implementation of `newton` with linesearch using golden section with Laplace as a constant
#   approximation of the hessian

# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_SFD",
                    "minimizer": ["scipy_L-BFGS-B", "scipy_Newton-CG_hessp"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_Laplace",
                    "minimizer": ["my_newton"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")


# %% [markdown]
# # Solvers with only grad information
# ## Comparison of implementations (numpy vs numba vs jax)

# %%
all_solvers = []
all_solvers.append({"val_grad": "numpy",
                    "hess": "numpy_Laplace",
                    "minimizer": ["scipy_TNC"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_Laplace",
                    "minimizer": ["scipy_TNC"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_Laplace",
                    "minimizer": ["scipy_TNC"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")


# %% [markdown]
# ## For numba a comparison of multiple minimizators

# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_Laplace",
                    "minimizer": ["scipy_TNC", "scipy_BFGS", "scipy_CG", "scipy_L-BFGS-B"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")


# %% [markdown]
# # Solvers using hessians
# **in scipy**
# - `trust-exact` needs dense matrix, no reasonable warkaround
# - `newton-cg` needs hessian-vector product
#
# **own implementation of trust region method**
#
# ## Comparison of multiple implementation of Hessian approximation
# - exact from jax autodiff - both dense and sparse version
# - approx using SDF (copy of matlab code)
# - constant approximation using Laplace (p=2)

# %%
all_solvers = []
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_dense",
                    "minimizer": ["scipy_trust-exact", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_sparse",
                    "minimizer": ["my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_SFD",
                    "minimizer": ["my_trust_region", "scipy_Newton-CG_hessp"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_Laplace",
                    "minimizer": ["my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")


# %% [markdown]
# # Comparison of implementations (numpy vs numba vs jax)
#  - for trust region
#  - also for newton method as both SFD and Laplace converge

# %%
all_solvers = []
all_solvers.append({"val_grad": "numpy",
                    "hess": "numpy_Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numpy",
                    "hess": "numpy_SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")


# %% [markdown]
# # Initial guess as solution of Laplace (p=2)
# Exact hessian (in jax autodiff) very much strougle when starting from 0. It is in orders of magnitude faster
# when starting from initial guess as solution of Laplace (p=2) (which is cheap in comparison).

# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_Laplace",
                    "minimizer": ["my_newton"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_Laplace",
                    "minimizer": ["my_newton"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "numba_SFD",
                    "minimizer": ["my_newton"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_SFD",
                    "minimizer": ["my_newton"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "jax",
                    "hess": "jax_sparse",
                    "minimizer": ["my_newton"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, compile_df, f_val_df, solve_df = test_runner_1D.create_tables(results, display_table="python")
