from jax import config
from numba import jit
from scipy.sparse import diags  # type: ignore
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.sparse.linalg._dsolve.linsolve import factorized
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
import warnings

import os
import sys

# add .. to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import sparsejac  # noqa
from tools import hessian_tools  # noqa


config.update("jax_enable_x64", True)


@jit(nopython=True)
def _energy_numba(v_internal, f_mid, v, h, p):
    v[1:-1] = v_internal
    n = h.shape[0]
    Jv_density = 0.0
    for i in range(n):
        Jv_density += h[i] * ((1 / p) * np.abs((v[i + 1] - v[i]) / h[i])**p - f_mid[i] * (v[i + 1] + v[i]) / 2)
    return Jv_density


def _energy_jax(v_internal, fx, v, h, p):
    v = v.at[1:-1].set(v_internal)
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * jnp.abs(vx)**p - fx * v_mid

    return jnp.sum(h * Jv_density)


def _energy_numpy(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * np.abs(vx)**p - fx * v_mid
    return np.sum(h * Jv_density)


def ff_mass_construct(ff_full, n, h):

    # Length of diagonal elements is n+2
    # Length of sub-diagonal and super-diagonal elements is n+1
    diagonal_elems = np.arange(n + 1)
    h_single = h[0]
    # Rows and columns indices for the diagonals
    row_diag = np.concatenate([diagonal_elems, diagonal_elems[1:], diagonal_elems[:-1]])
    col_diag = np.concatenate([diagonal_elems, diagonal_elems[:-1], diagonal_elems[1:]])

    # Data for the diagonals
    # [[h/3 2*h*ones(1,n)/3 h/3] h*ones(1,2*n+2)/6]
    data_diag = np.concatenate([np.array([h_single / 3]), 2 * h_single / 3 * np.ones(n - 1),
                                np.array([h_single / 3]), h_single / 6 * np.ones(2 * n)])

    Mass = sp.coo_matrix((data_diag, (row_diag, col_diag)), shape=(n + 1, n + 1))  # type: ignore

    ff_mass = ff_full @ Mass

    return ff_mass


def _energy_numpy_grad(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    tmp1 = np.sign(v[1:-1] - v[:-2]) * (np.abs(v[1:-1] - v[:-2]) / h[:-1]) ** (p - 1)
    tmp2 = np.sign(v[2:] - v[1:-1]) * (np.abs(v[2:] - v[1:-1]) / h[1:]) ** (p - 1)
    return tmp1 - tmp2 - fx[1:-1]


@jit(nopython=True)
def _energy_numba_grad(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    n = h.shape[0]
    tmp1 = np.zeros(n - 1)
    tmp2 = np.zeros(n - 1)
    for i in range(n - 1):
        tmp1[i] = np.sign(v[i + 1] - v[i]) * (np.abs(v[i + 1] - v[i]) / h[i]) ** (p - 1)
        tmp2[i] = np.sign(v[i + 2] - v[i + 1]) * (np.abs(v[i + 2] - v[i + 1]) / h[i + 1]) ** (p - 1)
    return tmp1 - tmp2 - fx[1:-1]


# rhs
def f_default(x):
    return -10 * np.ones(x.size)


def sparse_tridiagonal_scipy(n):
    """Create a three-diagonal CSR matrix with ones on its diagonals.

    Args:
    n: The number of rows (or columns) in the matrix.

    Returns:
    The created matrix as a CSR matrix.
    """
    # Create the diagonals
    diagonals = np.ones((3, n))
    # The main diagonal has offset 0, the diagonals above and below it have offsets -1 and 1, respectively
    offsets = [-1, 0, 1]
    # Use the `diags` function to create a sparse matrix with these diagonals
    matrix = diags(diagonals, offsets, shape=(n, n), format='csr')
    return matrix


def laplace_1D(x):
    # Calculate distances between nodes
    h = np.diff(x)

    # Initialize diagonals
    main_diag = np.hstack((1, 1 / h[:-1] + 1 / h[1:], 1))
    lower_diag = np.hstack((0, -1 / h[1:]))
    upper_diag = np.hstack((-1 / h[:-1], 0))

    # Create the sparse matrix
    laplace_matrix = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1]).tocsc()

    return laplace_matrix[1:-1, 1:-1]


def sparse_tridiagonal_jax(n):
    # Prepare row and column indices
    main_diag = jnp.arange(n)
    off_diag = jnp.arange(n - 1)

    # Indices for main diagonal
    rows_main = main_diag
    cols_main = main_diag

    # Indices for the diagonal above the main diagonal
    rows_above = off_diag
    cols_above = off_diag + 1

    # Indices for the diagonal below the main diagonal
    rows_below = off_diag + 1
    cols_below = off_diag

    # Concatenate all indices
    rows = jnp.concatenate([rows_main, rows_above, rows_below])
    cols = jnp.concatenate([cols_main, cols_above, cols_below])

    # Combine rows and cols into a single 2D indices array
    indices = jnp.stack([rows, cols], axis=-1)

    # Create data array with ones
    data = jnp.ones_like(rows)

    # Create sparse matrix
    matrix = sparse.BCOO((data, indices), shape=(n, n))

    return matrix


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
            solution = spla.spsolve(self.H, x)  # type: ignore
            return solution
        except np.linalg.LinAlgError:
            # If a LinAlgError is raised (which includes singular matrix), return x
            print("LinAlgError")
            return x
        except spla.MatrixRankWarning:  # type: ignore
            # If a MatrixRankWarning is raised (which includes singular matrix), return x
            print("MatrixRankWarning")
            return x

    def solve_trust(self, x, c):
        return spla.spsolve(self.H + sp.eye(self.H.shape[0]) * c, x)  # type: ignore

    def norm(self, x, c):
        return x @ (self.H + sp.eye(self.H.shape[0]) * c) @ x  # type: ignore


class HessGenerator:
    def __init__(self, ddf):
        self.ddf = ddf

    def __call__(self, x):
        return HessSolve(self.ddf(x))


class HessGeneratorSparseJax(HessGenerator):
    def __call__(self, x):
        H_jax = self.ddf(x)
        data_np = np.array(H_jax.data)
        indices_np = np.array(H_jax.indices)
        shape_np = H_jax.shape
        sparse_matrix_scipy = sp.csr_matrix(  # type: ignore
            (data_np, (indices_np[:, 0], indices_np[:, 1])), shape=shape_np)
        return HessSolveSparse(sparse_matrix_scipy)


class HessGeneratorSparse(HessGenerator):
    def __call__(self, x):
        sparse_matrix_scipy = self.ddf(x)
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
        self.H_inv = factorized(H)

    def __call__(self, x):
        return HessSolverLaplace(self.H, self.H_inv)


class JaxEnergy:
    def __init__(self, p, a, b, ne, f=f_default):
        self.ne = ne
        self.p = p
        self.a = a
        self.b = b
        self.f = f

        self.x = jnp.linspace(self.a, self.b, self.ne + 1, dtype=jnp.float64)
        self.x_mid = (self.x[1:] + self.x[:-1]) / 2
        self.h = jnp.diff(self.x)
        self.v = jnp.zeros_like(self.x, dtype=jnp.float64)
        self.v_internal = self.v[1:-1].copy()
        self.x0 = self.v_internal.copy()   # initial guess
        self.fx = self.f(self.x_mid)

    def change_problem(self, p=None, a=None, b=None, ne=None, f=None):
        if p is not None:
            self.p = p
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if ne is not None:
            self.ne = ne
        if f is not None:
            self.f = f
        self.x = jnp.linspace(self.a, self.b, self.ne + 1, dtype=jnp.float64)
        self.x_mid = (self.x[1:] + self.x[:-1]) / 2
        self.h = jnp.diff(self.x)
        self.v = jnp.zeros_like(self.x, dtype=jnp.float64)
        self.v_internal = self.v[1:-1].copy()
        self.x0 = self.v_internal.copy()   # initial guess
        self.fx = self.f(self.x_mid)

    def recompile(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        ddfun = jax.jit(jax.hessian(_energy_jax, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        def dfun1(v_internal): return dfun(v_internal, self.fx, self.v, self.h, self.p)
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, HessGenerator(ddfun1)


class JaxEnergySparse(JaxEnergy):
    def recompile(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        # create three diagonal matrix
        sparsity = sparse_tridiagonal_jax(self.ne - 1)
        ddfun = jax.jit(sparsejac.jacrev(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        def dfun1(v_internal): return dfun(v_internal, self.fx, self.v, self.h, self.p)
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, HessGeneratorSparseJax(ddfun1)


class JaxEnergyLaplace(JaxEnergy):
    def recompile(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        def dfun1(v_internal): return dfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        H = laplace_1D(self.x)
        return fun1, dfun1, HessGeneratorLaplace(H)


class JaxEnergySparseLaplace(JaxEnergy):
    def recompile(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        # create three diagonal matrix
        sparsity = sparse_tridiagonal_jax(self.ne - 1)
        ddfun = jax.jit(sparsejac.jacrev(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        def dfun1(v_internal): return dfun(v_internal, self.fx, self.v, self.h, self.p)
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        _ = ddfun1(self.v_internal)
        H = laplace_1D(self.x)
        return fun1, dfun1, HessGeneratorSparseJax(ddfun1), HessGeneratorLaplace(H)


class NumpyEnergy:
    def __init__(self, p, a, b, ne, f=f_default):
        self.ne = ne
        self.p = p
        self.a = a
        self.b = b
        self.f = f

        self.x = np.linspace(self.a, self.b, self.ne + 1)
        self.x_mid = (self.x[1:] + self.x[:-1]) / 2
        self.h = np.diff(self.x)
        self.v = np.zeros_like(self.x)
        self.v_internal = self.v[1:-1].copy()
        self.x0 = self.v_internal.copy()
        self.fx = self.f(self.x_mid)

    def change_problem(self, p=None, a=None, b=None, ne=None, f=None):
        if p is not None:
            self.p = p
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if ne is not None:
            self.ne = ne
        if f is not None:
            self.f = f
        self.x = np.linspace(self.a, self.b, self.ne + 1)
        self.x_mid = (self.x[1:] + self.x[:-1]) / 2
        self.h = np.diff(self.x)
        self.v = np.zeros_like(self.x)
        self.v_internal = self.v[1:-1].copy()
        self.x0 = self.v_internal.copy()
        self.fx = self.f(self.x_mid)

    def recompile(self):
        pass


class NumpyEnergySFD(NumpyEnergy):
    def recompile(self):
        def fun1(v_internal): return _energy_numpy(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numpy_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        Hstr = sparse_tridiagonal_scipy(self.ne - 1)
        groups = hessian_tools.color(Hstr)
        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, groups, dfun1)
        return fun1, dfun1, HessGeneratorSparse(ddfun1)


class NumbaEnergySFD(NumpyEnergy):
    def recompile(self):
        def fun1(v_internal): return _energy_numba(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numba_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        Hstr = sparse_tridiagonal_scipy(self.ne - 1)
        groups = hessian_tools.color(Hstr)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, groups, dfun1)
        return fun1, dfun1, HessGeneratorSparse(ddfun1)


class NumpyEnergyLaplace(NumpyEnergy):
    def recompile(self):
        def fun1(v_internal): return _energy_numpy(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numpy_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        H = laplace_1D(self.x)
        return fun1, dfun1, HessGeneratorLaplace(H)


class NumbaEnergyLaplace(NumpyEnergy):
    def recompile(self):
        def fun1(v_internal): return _energy_numba(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numba_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        H = laplace_1D(self.x)
        return fun1, dfun1, HessGeneratorLaplace(H)


class JaxEnergySFD(JaxEnergy):
    def recompile(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        def dfun1(v_internal): return dfun(v_internal, self.fx, self.v, self.h, self.p)
        Hstr = sparse_tridiagonal_scipy(self.ne - 1)
        self.groups = hessian_tools.color(Hstr)
        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, self.groups, dfun1)
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, HessGeneratorSparse(ddfun1)
