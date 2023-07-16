from jax import config
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from .suplement_sparse_matrices import laplace_1D, sparse_tridiagonal_scipy

import os
import sys

# add .. to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import sparsejac  # noqa
from tools import hessian_tools  # noqa


config.update("jax_enable_x64", True)


def _energy_jax(v_internal, fx, v, h, p):
    # v = v.at[1:-1].set(v_internal)
    v = jnp.concatenate([v[:1], v_internal, v[-1:]], axis=0)
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * jnp.abs(vx)**p - fx * v_mid

    return jnp.sum(h * Jv_density)


# rhs
def f_default(x):
    return -10 * np.ones(x.size)


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


class HessGeneratorSparseJax(hessian_tools.HessGenerator):
    def __call__(self, x):
        H_jax = self.ddf(x)
        data_np = np.array(H_jax.data)
        indices_np = np.array(H_jax.indices)
        shape_np = H_jax.shape
        sparse_matrix_scipy = sp.csr_matrix(  # type: ignore
            (data_np, (indices_np[:, 0], indices_np[:, 1])), shape=shape_np)
        return hessian_tools.HessSolveSparse(sparse_matrix_scipy)


class Energy:
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

    def _only_grad_val(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))
        def fun1(v_internal): return fun(v_internal, self.fx, self.v, self.h, self.p)
        fun = jax.jit(fun)
        dfun = jax.jit(dfun)
        def dfun1(v_internal): return np.asfortranarray(np.array(dfun(v_internal, self.fx, self.v, self.h, self.p)))
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
        return fun1, dfun1

    def recompile(self):
        fun1, dfun1 = self._only_grad_val()
        return fun1, dfun1, None


class EnergyDense(Energy):
    def recompile(self):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        ddfun = jax.jit(jax.hessian(_energy_jax, argnums=0))
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, hessian_tools.HessGenerator(ddfun1)


class EnergySparse(Energy):
    def recompile(self):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        # create three diagonal matrix
        sparsity = sparse_tridiagonal_jax(self.ne - 1)
        ddfun = jax.jit(sparsejac.jacrev(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0))
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, HessGeneratorSparseJax(ddfun1)


class EnergyLaplace(Energy):
    def recompile(self):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        H = laplace_1D(self.x)
        return fun1, dfun1, hessian_tools.HessGeneratorLaplace(H)


class EnergySparseLaplace(Energy):
    def recompile(self):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        # create three diagonal matrix
        sparsity = sparse_tridiagonal_jax(self.ne - 1)
        ddfun = jax.jit(sparsejac.jacrev(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0))
        def ddfun1(v_internal): return ddfun(v_internal, self.fx, self.v, self.h, self.p)
        _ = ddfun1(self.v_internal)
        H = laplace_1D(self.x)
        return fun1, dfun1, HessGeneratorSparseJax(ddfun1), hessian_tools.HessGeneratorLaplace(H)


class EnergySFD(Energy):
    def recompile(self):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        Hstr = sparse_tridiagonal_scipy(self.ne - 1)
        self.groups = hessian_tools.color(Hstr)
        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, self.groups, dfun1)
        _ = ddfun1(self.v_internal)
        return fun1, dfun1, hessian_tools.HessGeneratorSparse(ddfun1)
