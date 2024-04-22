from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore
import pyamg
import igraph
import numpy as np
import scipy.sparse as sp  # type: ignore
import warnings
import scipy.sparse.linalg as spla  # type: ignore
from scipy.sparse.linalg._dsolve.linsolve import factorized  # type: ignore


class HessSolve:
    def __init__(self, H):
        self.H = H

    def solve(self, x):
        try:
            return np.linalg.solve(self.H, x)
        except np.linalg.LinAlgError:
            pass
        return x

    def solve_trust(self, x, c):
        return np.linalg.solve(self.H + np.eye(self.H.shape[0]) * c, x)

    def norm(self, x, c):
        return x @ (self.H + np.eye(self.H.shape[0]) * c) @ x


class HessSolveSparse(HessSolve):
    def solve(self, x):
        # try to solve with scipy (spla.spsolve(self.H, x)) but if matrix is singular, return x
        warnings.filterwarnings("error")
        try:
            # Try to solve the system
            x = spla.spsolve(self.H, x.copy())  # type: ignore
        except np.linalg.LinAlgError:
            # If a LinAlgError is raised (which includes singular matrix), return x
            print("LinAlgError")
        except spla.MatrixRankWarning:  # type: ignore
            # If a MatrixRankWarning is raised (which includes singular matrix), return x
            print("MatrixRankWarning")
        warnings.filterwarnings("default")
        return x

    def solve_trust(self, x, c):
        return spla.spsolve(self.H + sp.eye(self.H.shape[0], format="csc") * c, x)  # type: ignore

    def norm(self, x, c):
        return x @ (self.H + sp.eye(self.H.shape[0]) * c) @ x  # type: ignore


class HessSolveSparseIterative(HessSolveSparse):
    def __init__(self, H, verbose=False, tol=1e-3, maxiter=5000):
        self.H = H
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, x):
        # conjugate gradient solution of self.H with  x
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        # Extract diagonal and create diagonal preconditioner
        D_inv = 1.0 / np.abs(self.H.diagonal())
        M = sp.diags(D_inv)
        # Convert M to a linear operator
        def M_op(x): return M.dot(x)
        M_lin = spla.LinearOperator(M.shape, M_op)

        sol = spla.cg(self.H, x.copy(), tol=self.tol, M=M_lin, callback=callback, maxiter=self.maxiter)[0]

        if self.verbose:
            print(f"Iterations in CG solver: {iteration_count[0]}.")

        return sol

    def solve_trust(self, x, c):
        H = self.H + sp.eye(self.H.shape[0], format="csr") * c
        # conjugate gradient solution of self.H with  x
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        # Extract diagonal and create diagonal preconditioner
        D_inv = 1.0 / np.abs(H.diagonal())
        M = sp.diags(D_inv)
        # Convert M to a linear operator
        def M_op(x): return M.dot(x)
        M_lin = spla.LinearOperator(M.shape, M_op)

        sol = spla.cg(H, x.copy(), tol=self.tol, M=M_lin, callback=callback, maxiter=self.maxiter)[0]

        if self.verbose:
            print(f"Iterations in CG solver: {iteration_count[0]}.")

        return sol


class HessSolveSparseAMG(HessSolveSparse):
    def __init__(self, H, verbose=False, tol=1e-3, maxiter=1000):
        self.H = H
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, x):
        # conjugate gradient solution of self.H with  x
        ml = pyamg.ruge_stuben_solver(self.H.tocsr())
        M_lin = ml.aspreconditioner()

        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        sol = spla.cg(self.H, x.copy(), tol=self.tol, M=M_lin, callback=callback, maxiter=self.maxiter)[0]
        if self.verbose:
            print(f"Iterations in AMG solver: {iteration_count[0]}.")

        return sol

    def solve_trust(self, x, c):
        H = self.H + sp.eye(self.H.shape[0], format="csr") * c
        ml = pyamg.ruge_stuben_solver(H.tocsr())
        M_lin = ml.aspreconditioner()
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        sol = spla.cg(H, x.copy(), tol=self.tol, M=M_lin, callback=callback, maxiter=self.maxiter)[0]
        if self.verbose:
            print(f"Iterations in AMG solver: {iteration_count[0]}.")

        return sol


class HessGenerator:
    def __init__(self, ddf):
        self.ddf = ddf

    def __call__(self, x):
        return HessSolve(self.ddf(x))


class HessGeneratorSparse(HessGenerator):
    def __init__(self, ddf, solver="direct", verbose=False, tol=1e-3, maxiter=5000):
        self.ddf = ddf
        self.solver = solver
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, x):
        sparse_matrix_scipy = self.ddf(x)
        if self.solver == "direct":
            return HessSolveSparse(sparse_matrix_scipy)
        elif self.solver == "cg":
            return HessSolveSparseIterative(sparse_matrix_scipy, verbose=self.verbose, tol=self.tol,
                                            maxiter=self.maxiter)
        elif self.solver == "amg":
            return HessSolveSparseAMG(sparse_matrix_scipy, verbose=self.verbose, tol=self.tol,
                                      maxiter=self.maxiter)
        else:
            raise ValueError("Unknown solver")


class HessSolverLaplace:
    def __init__(self, H, H_inv):
        self.H = H
        self.H_inv = H_inv

    def solve(self, x):
        return self.H_inv(np.array(x))

    def solve_trust(self, x, c):
        return spla.spsolve(self.H + sp.eye(self.H.shape[0]) * c, x)  # type: ignore

    def norm(self, x, c):
        return x @ (self.H + sp.eye(self.H.shape[0]) * c) @ x  # type: ignore


class HessGeneratorLaplace:
    def __init__(self, H):
        self.H = H
        self.H_inv = factorized(H.tocsc())

    def __call__(self, x):
        return HessSolverLaplace(self.H, self.H_inv)


def color_old(J, p=None):
    m, n = J.shape
    if p is None:
        p = reverse_cuthill_mckee(J)

    J = J[:, p]
    group = np.zeros(n, dtype=int)
    ncol = 0

    while np.any(group == 0):
        ncol += 1
        rows = np.zeros(m)
        index = np.where(group == 0)[0]
        lenindex = len(index)
        for i in range(lenindex):
            k = index[i]
            inner = np.inner(J[:, k].todense().ravel(), rows)
            if inner == 0:
                group[k] = ncol
                rows += J[:, k].toarray().flatten()

    group[p] = group.copy()
    return group


def color(J):
    n = J.shape[0]
    J_2 = J @ J.T
    rows, cols = np.nonzero(J_2)
    edges = np.vstack((rows, cols)).T
    g = igraph.Graph()
    g.add_vertices(n)
    g.add_edges(edges)
    coloring = g.vertex_coloring_greedy()
    # coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')
    # group_nx = np.zeros(n, dtype=int)

    # for node, color in coloring.items():
    #     group_nx[node] = color + 1  # Add 1 to match the 1-based coloring of the original function
    return np.array(coloring) + 1


def sfd(x, grad, Hstr, group, dx):
    xcurr = x.flatten()
    m, n = Hstr.shape
    v = np.zeros(n)
    ncol = np.max(group)

    alpha = 6e-8 * np.ones(ncol)

    # H = lil_matrix(Hstr != 0, dtype=float)  # Equivalent of MATLAB's spones
    rowH, colH = Hstr.nonzero()
    all_new_rows = []
    all_new_cols = []
    all_new_data = []
    xnrm = 1  # max(np.linalg.norm(xcurr), 1) ** 2  # type: ignore
    for k in range(ncol):
        d = (group == k + 1)
        alpha[k] *= xnrm
        y = xcurr + alpha[k] * d

        v = dx(y)
        w = (v - grad) / alpha[k]
        mask_H_cols = d[colH]

        new_rows = rowH[mask_H_cols]
        new_cols = colH[mask_H_cols]
        new_data = w[new_rows]
        all_new_rows.append(new_rows)
        all_new_cols.append(new_cols)
        all_new_data.append(new_data)

    new_rows_singe_array = np.concatenate(all_new_rows + all_new_cols)
    new_cols_singe_array = np.concatenate(all_new_cols + all_new_rows)
    new_data_singe_array = np.concatenate(all_new_data + all_new_data) / 2  # type: ignore
    H = sp.csc_matrix((new_data_singe_array, (new_rows_singe_array, new_cols_singe_array)),  # type: ignore
                      shape=(m, n), copy=True)
    # print(alpha)
    H.sum_duplicates()
    return H
