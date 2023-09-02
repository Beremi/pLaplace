# add .. to path
import pandas as pd
import sys
import os
import time
from scipy.optimize._minimize import minimize
from IPython.display import display
import pyamg
import numpy as np
from . import mesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import minimizers  # noqa

# all possible types of inputs (not all combinations allowed)
_ = ["numpy", "numba", "jax", "none"]  # "val_grad"
_ = ["Laplace", "SFD", "dense", "sparse"]  # "hess"
_ = ["my_newton", "my_trust_region",
     "scipy_trust-exact", "scipy_Newton-CG", "scipy_BFGS", "scipy_L-BFGS-B"]  # "minimizer"
_ = ["zero", "laplace"]  # "initial_guess"


class MyHessP:
    def __init__(self, ddf):
        self.ddf = ddf
        self.x = None
        self.iters = []

    def __call__(self, x, p):
        if self.x is not x:
            self.iters.append(0)
            self.x = x
            self.H = self.ddf(x)
        self.iters[-1] += 1
        return self.H @ p


class HessScipyDense:
    def __init__(self, ddf):
        self.ddf = ddf

    def __call__(self, x):
        H = self.ddf.ddf(x)  # H is a sparse matrix
        return H  # Return the product of H and p


def create_tables(data, display_table="latex"):
    # suppress DepricationWarnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Create empty DataFrames
    iterations_df = pd.DataFrame()
    compile_df = pd.DataFrame()
    f_val_df = pd.DataFrame()
    solve_df = pd.DataFrame()
    combined_time_df = pd.DataFrame()
    combined_df = pd.DataFrame()
    combined_df2 = pd.DataFrame()

    for d in data:
        my_tuple = d['solver_id']
        # Remove quotes and join the elements with '/'
        my_tuple_str = '/'.join(str(item) for item in my_tuple)

        # Replace underscores with spaces
        my_tuple_str = my_tuple_str.replace('_', ' ')
        solver_id = str(my_tuple_str)  # Use string of tuple as index
        size = d['size']

        # Add values to respective DataFrames
        iterations_df.loc[solver_id, size] = "{:.0f}".format(d['iterations'])
        compile_df.loc[solver_id, size] = "{:.3f}".format(d['duration_compile'])
        f_val_df.loc[solver_id, size] = "{:.7f}".format(d['f_val'])
        solve_df.loc[solver_id, size] = "{:.3f}".format(d['duration_solve'])

    # Combine compile and solve times
    combined_time_df = compile_df.astype(str) + ' / ' + solve_df.astype(str)

    # Create combined DataFrame
    for col in combined_time_df.columns:
        combined_df[(col, 'time prep/sol')] = combined_time_df[col]
        combined_df[(col, 'iters')] = iterations_df[col]

    for col in solve_df.columns:
        combined_df2[(col, 'time')] = solve_df[col]
        combined_df2[(col, 'iters')] = iterations_df[col]

    # Fill NaN values with "-"
    iterations_df.fillna("-", inplace=True)
    compile_df.fillna("-", inplace=True)
    f_val_df.fillna("-", inplace=True)
    solve_df.fillna("-", inplace=True)
    combined_time_df.fillna("-", inplace=True)
    combined_df.fillna("-", inplace=True)
    combined_df2.fillna("-", inplace=True)

    # Make the columns a MultiIndex
    combined_df.columns = pd.MultiIndex.from_tuples(
        [(r"$n=" + str(col[0]) + r"$", col[1]) for col in combined_df.columns], names=[r'$\mathcal{N}_{dof}$', 'Option'])
    combined_df2.columns = pd.MultiIndex.from_tuples(
        [(r"$n=" + str(col[0]) + r"$", col[1]) for col in combined_df2.columns], names=[r'$|\mathcal{N}_{dof}|$', 'Option'])

    if display_table == "Ipython":
        print("Function value:")
        display(f_val_df)
        print("Time and iters:")
        display(combined_df)
    elif display_table == "python":
        print("Function value:")
        print(f_val_df)
        print("Time and iters:")
        print(combined_df)
    elif display_table == "latex":
        print("Function value:")
        print(f_val_df.to_latex(escape=False))
        print("Time and iters:")
        print(combined_df.to_latex(escape=False))

    return iterations_df, f_val_df, compile_df, solve_df, combined_time_df, combined_df, combined_df2


def test_runner_2D(list_of_tests):
    results = []
    mesh_obj = mesh.Mesh2D()
    mesh_obj.load_from_matfile('pLaplace_2D/data_export_matlab/mesh_2D_level1.mat')
    for test in list_of_tests:
        val_grad = test["val_grad"]
        hess = test["hess"]
        solver_type = test["solver_type"]
        problem_setting = test["problem_setting"]
        p = problem_setting["p"]

        if val_grad == "numpy":
            from . import energy_numpy as energy
        elif val_grad == "numba":
            from . import energy_numba as energy
        elif val_grad == "jax":
            from . import energy_jax as energy
        else:
            raise ValueError("val_grad not recognized")

        if hess == "Laplace":
            raise ValueError("hess not recognized")
        elif hess == "SFD":
            energy_constructor = energy.EnergySFD(mesh_obj, p)
        elif hess == "dense" and val_grad == "jax":
            raise ValueError("hess not recognized")
        elif hess == "sparse" and val_grad == "jax":
            energy_constructor = energy.EnergySparse(mesh_obj, p)  # type: ignore
        elif hess == "none":
            energy_constructor = energy.Energy(mesh_obj, p)
        else:
            raise ValueError("hess not recognized")

        for size in test["sizes"]:
            size_name = f'pLaplace_2D/data_export_matlab/mesh_2D_level{size}.mat'
            mesh_obj = mesh.Mesh2D()
            mesh_obj.load_from_matfile(size_name)

            # _ = [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]

            start_time = time.time()
            energy_constructor.change_problem(mesh_obj, p)
            out = energy_constructor.recompile(solver=solver_type, verbose=False)  # type: ignore
            end_time = time.time()
            duration_compile = end_time - start_time

            f = out[0]
            df = out[1]
            ddf = out[2]
            if test["initial_guess"] == "zero":
                x0 = energy_constructor.u.copy()
            elif test["initial_guess"] == "rand":
                x0 = np.random.rand(energy_constructor.u.shape[0])  # type: ignore
            elif test["initial_guess"] == "laplace":
                from .suplement_matrices import stiffness_matrix
                A_stiff = stiffness_matrix(mesh_obj)[mesh_obj.free_nodes, :][:, mesh_obj.free_nodes]  # type: ignore
                ml = pyamg.ruge_stuben_solver(A_stiff)
                x0 = ml.solve(energy_constructor.fx[mesh_obj.free_nodes], tol=1e-6, maxiter=200)
            else:
                raise ValueError("initial_guess not recognized")

            for minimizer in test["minimizer"]:
                if minimizer == "my_newton":
                    def minimizer_func(f, df, ddf, x0): return minimizers.newton2(  # type: ignore
                        f, df, ddf, x0, tolf=1e-9)
                elif minimizer == "my_newton-doubledir":
                    # throw not implemented error
                    raise ValueError("Not implemented")
                elif minimizer == "my_trust_region":
                    def minimizer_func(f, df, ddf, x0): return minimizers.trust_region2(  # type: ignore
                        f, df, ddf, x0, tolf=1e-9, maxit=10000)
                elif minimizer == "scipy_trust-exact":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hess=HessScipyDense(ddf),
                                                                        method='trust-exact', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_trust-ncg":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hess=HessScipyDense(ddf),
                                                                        method='trust-ncg', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_trust-krylov":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hess=HessScipyDense(ddf),
                                                                        method='trust-krylov', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_dogleg":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hess=HessScipyDense(ddf),
                                                                        method='dogleg', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_Newton-CG_hessp":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hessp=MyHessP(ddf.ddf),
                                                                        method='Newton-CG', tol=1e-6, options={'maxiter': 1000, 'xtol': 1e-9})
                elif minimizer == "scipy_trust-krylov_hessp":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hessp=MyHessP(ddf.ddf),
                                                                        method='trust-krylov', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_trust-ncg_hessp":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, hessp=MyHessP(ddf.ddf),
                                                                        method='trust-ncg', tol=1e-6, options={'maxiter': 1000})
                elif minimizer == "scipy_Newton-CG":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df,
                                                                        method='Newton-CG', tol=1e-9, options={'maxiter': 10000})
                elif minimizer == "scipy_SLSQP":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df,
                                                                        method='SLSQP', tol=1e-9, options={'maxiter': 10000})
                elif minimizer == "scipy_BFGS":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df,
                                                                        method='BFGS', tol=1e-9, options={'maxiter': 1000})
                elif minimizer == "scipy_L-BFGS-B":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df,
                                                                        method='L-BFGS-B', tol=1e-13, options={'maxiter': 100000, 'maxfun': 10000000})
                elif minimizer == "scipy_CG":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df,
                                                                        method='CG', tol=1e-9, options={'maxiter': 100000})
                elif minimizer == "scipy_TNC":
                    def minimizer_func(f, df, ddf, x0): return minimize(f, x0, jac=df, method='TNC', tol=1e-9)
                else:
                    raise ValueError("minimizer not recognized")

                start_time = time.time()
                result = minimizer_func(f, df, ddf, x0)
                end_time = time.time()
                duration_solve = end_time - start_time
                solver_id = (val_grad, hess, minimizer, test["initial_guess"], solver_type)
                results.append({"solver_id": solver_id, "size": mesh_obj.free_nodes.shape[0], "duration_compile": duration_compile, "duration_solve": duration_solve,
                                "iterations": result.nit, "f_val": result.fun, "message": result.message})

    return results
