from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore
import numpy as np
import scipy.sparse as sp  # type: ignore
import warnings
import scipy.sparse.linalg as spla  # type: ignore
from scipy.sparse.linalg._dsolve.linsolve import factorized  # type: ignore
import networkx as nx


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


class HessGenerator:
    def __init__(self, ddf):
        self.ddf = ddf

    def __call__(self, x):
        return HessSolve(self.ddf(x))


class HessGeneratorSparse(HessGenerator):
    def __call__(self, x):
        sparse_matrix_scipy = self.ddf(x).tocsc()
        return HessSolveSparse(sparse_matrix_scipy)


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
    G = nx.from_scipy_sparse_array(J @ J)
    coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')
    group_nx = np.zeros(n, dtype=int)

    for node, color in coloring.items():
        group_nx[node] = color + 1  # Add 1 to match the 1-based coloring of the original function
    return group_nx


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
    new_data_singe_array = np.concatenate(all_new_data + all_new_data) / 2
    H = sp.csc_matrix((new_data_singe_array, (new_rows_singe_array, new_cols_singe_array)),  # type: ignore
                      shape=(m, n))
    # print(alpha)
    return H
