# %%
from pLaplace_1D import test_runner_1D

# all possible types of inputs (not all combinations allowed)
_ = ["numpy", "numba", "jax", "none"]  # "val_grad"
_ = ["Laplace", "SFD", "dense", "sparse"]  # "hess"
_ = ["my_newton", "my_trust_region",
     "scipy_trust-exact", "scipy_Newton-CG", "scipy_BFGS", "scipy_L-BFGS-B"]  # "minimizer"
_ = ["zero", "laplace"]  # "initial_guess"

problem_setting = {"p": 3,   # p in pLaplace
                   "a": -1,  # left bound of computational domain
                   "b": 1,   # right bound of computational domain
                   "f": "default"}  # source term, constant 10

sizes = [10, 100, 1000]


# %% [markdown]
# **-16.865480854231357**

# %% [markdown]
# # Conclusions first
# - fastest value and grad evaluation have numba (rewrite of matlab code)
# - fastest method without the use of user provided hessian is `L-BFGS-B` (scipy)
# - method similar to the "best" in the matlab code is `Newton-CG` (scipy) with approximation of hessian using numrical differentiation SFD (rewrite of matlab code)
# - fastest method overall is my implementation of `newton` with linesearch using golden section with Laplace as a constant approximation of the hessian

# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "none",
                    "minimizer": ["scipy_L-BFGS-B"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["scipy_Newton-CG_hessp"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "Laplace",
                    "minimizer": ["my_newton"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")


# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")


# %% [markdown]
# # Solvers with only grad information
# ## Comparison of implementations (numpy vs numba)

# %%
all_solvers = []
all_solvers.append({"val_grad": "numpy",
                    "hess": "none",
                    "minimizer": ["scipy_L-BFGS-B"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "none",
                    "minimizer": ["scipy_L-BFGS-B"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")


# %% [markdown]
# ## For numba a comparison of multiple minimizators

# %%
all_solvers = []
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["scipy_TNC", "scipy_BFGS", "scipy_CG", "scipy_L-BFGS-B"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")


# %% [markdown]
# # Solvers using hessians
# **in scipy**
# - `trust-exact` needs dense matrix, no reasonable warkaround
# - `newton-cg` needs hessian-vector product
#
# **own implementation of trust region method**
#
# ## Comparison of multiple implementation of Hessian approximation
# - approx using SDF (copy of matlab code)
# - constant approximation using Laplace (p=2)

# %% [markdown]
# # Comparison of implementations (numpy vs numba)
#  - for trust region
#  - also for newton method as both SFD and Laplace converge

# %%
all_solvers = []
all_solvers.append({"val_grad": "numpy",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numpy",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "zero",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")


# %% [markdown]
# # Initial guess as solution of Laplace (p=2)
# Exact hessian (in jax autodiff) very much strougle when starting from 0. It is in orders of magnitude faster when starting
# from initial guess as solution of Laplace (p=2) (which is cheap in comparison).

# %%
all_solvers = []
all_solvers.append({"val_grad": "numpy",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "Laplace",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numpy",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
all_solvers.append({"val_grad": "numba",
                    "hess": "SFD",
                    "minimizer": ["my_newton", "my_trust_region"],
                    "initial_guess": "laplace",
                    "problem_setting": problem_setting,
                    "sizes": sizes})
results = test_runner_1D.test_runner_1D(all_solvers)
iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2 = test_runner_1D.create_tables(
    results, display_table="latex")
