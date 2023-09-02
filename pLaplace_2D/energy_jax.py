import scipy.sparse as sp
from jax.experimental import sparse

import jax
from jax import config
import jax.numpy as jnp
from pLaplace_2D.mesh import Mesh2D
from pLaplace_2D.suplement_matrices import mass_matrix, incidence_matrix
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import hessian_tools  # noqa
from tools import sparsejac  # noqa
config.update("jax_enable_x64", True)


def _energy_jax(u, u_0, free_nodes, elements, gradx, grady, p, f, areas):
    v = jnp.array(u_0)
    v = v.at[free_nodes].set(u)

    v_elems = v[elements]
    v_x_elems = jnp.sum(gradx * v_elems, axis=1) ** 2
    v_y_elems = jnp.sum(grady * v_elems, axis=1) ** 2
    intgrds1 = (1 / p) * (v_x_elems + v_y_elems) ** (p/2)
    e = jnp.sum(areas * intgrds1) - jnp.dot(f, v)
    return e


class HessGeneratorSparseJax(hessian_tools.HessGenerator):
    def __init__(self, ddf, solver="direct", verbose=False, tol=1e-3, maxiter=5000):
        self.ddf = ddf
        self.last_sol = None
        self.solver = solver
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, x):
        H_jax = self.ddf(x)
        data_np = np.array(H_jax.data)
        indices_np = np.array(H_jax.indices)
        shape_np = H_jax.shape
        sparse_matrix_scipy = sp.csr_matrix((data_np, (indices_np[:, 0], indices_np[:, 1])), shape=shape_np)
        if self.solver == "direct":
            return hessian_tools.HessSolveSparse(sparse_matrix_scipy)
        elif self.solver == "cg":
            return hessian_tools.HessSolveSparseIterative(sparse_matrix_scipy, verbose=self.verbose, tol=self.tol,
                                                          maxiter=self.maxiter)
        elif self.solver == "amg":
            return hessian_tools.HessSolveSparseAMG(sparse_matrix_scipy, verbose=self.verbose, tol=self.tol,
                                                    maxiter=self.maxiter)
        else:
            raise ValueError("Unknown solver")


class Energy:
    def __init__(self, mesh: Mesh2D, p=3, f=None):
        self.change_problem(mesh, p, f)

    def change_problem(self, mesh: Mesh2D, p=3, f=None):
        if f is None:
            f = np.ones((mesh.nodes.shape[0])) * (-10)
        self.p = p
        self.mesh = mesh

        self.M = mass_matrix(self.mesh)
        self.fx = self.M @ f
        self.u_0 = np.zeros_like(self.fx)

        self.u_0_jax = jnp.array(self.u_0, dtype=jnp.float64)
        self.free_nodes_jax = jnp.array(self.mesh.free_nodes, dtype=int)
        self.elements_jax = jnp.array(self.mesh.elements, dtype=int)
        self.gradx_jax = jnp.array(self.mesh.gradx, dtype=jnp.float64)
        self.grady_jax = jnp.array(self.mesh.grady, dtype=jnp.float64)

        self.fx_jax = jnp.array(self.fx, dtype=jnp.float64).ravel()
        self.areas_jax = jnp.array(self.mesh.areas, dtype=jnp.float64).ravel()
        self.u_jax = jnp.zeros_like(self.free_nodes_jax, dtype=jnp.float64)
        self.u = np.zeros_like(self.free_nodes_jax, dtype=np.float64)

    def _only_grad_val(self):
        # automatická derivace a kompilace
        fun = jax.jit(_energy_jax)
        dfun = jax.jit(jax.grad(_energy_jax, argnums=0))

        def fun1(v_internal): return fun(v_internal, self.u_0_jax, self.free_nodes_jax, self.elements_jax,
                                         self.gradx_jax, self.grady_jax, self.p, self.fx_jax, self.areas_jax)

        def dfun1(v_internal): return np.asfortranarray(np.array(dfun(v_internal, self.u_0_jax,
                                                                      self.free_nodes_jax, self.elements_jax,
                                                                      self.gradx_jax, self.grady_jax, self.p,
                                                                      self.fx_jax, self.areas_jax)))
        _ = fun1(self.u_jax)
        _ = dfun1(self.u_jax)
        return fun1, dfun1

    def recompile(self, solver="cg", verbose=False, tol=1e-3, maxiter=5000):
        fun1, dfun1 = self._only_grad_val()
        return fun1, dfun1, None


class EnergySparse(Energy):
    def recompile(self, solver="cg", verbose=False, tol=1e-3, maxiter=5000):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        # create three diagonal matrix

        sparsity_pattern_scipy = incidence_matrix(self.mesh)[self.mesh.free_nodes, :][:, self.mesh.free_nodes]

        # get indices of non-zero elements
        nonzero_indices = np.nonzero(sparsity_pattern_scipy)  # type: ignore

        # create sparse matrix in jax
        rows = nonzero_indices[0]
        cols = nonzero_indices[1]
        n = self.mesh.free_nodes.shape[0]

        indices = jnp.stack([rows, cols], axis=-1)

        # Create data array with ones
        data = jnp.ones_like(rows)

        # Create sparse matrix
        sparsity = sparse.BCOO((data, indices), shape=(n, n))
        self.ddfun = sparsejac.jacfwd(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0)
        ddfun = self.ddfun
        # ddfun = jax.jit(sparsejac.jacfwd(jax.grad(_energy_jax, argnums=0), sparsity, argnums=0))

        def ddfun1(v_internal): return ddfun(v_internal, self.u_0_jax, self.free_nodes_jax, self.elements_jax,
                                             self.gradx_jax, self.grady_jax, self.p, self.fx_jax, self.areas_jax)

        _ = ddfun1(self.u_jax)
        return fun1, dfun1, HessGeneratorSparseJax(ddfun1, solver=solver, verbose=verbose, tol=tol,
                                                   maxiter=maxiter)


class EnergySFD(Energy):
    def recompile(self, solver="cg", verbose=False, tol=1e-3, maxiter=5000):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        # create three diagonal matrix

        Hstr = incidence_matrix(self.mesh)[self.mesh.free_nodes, :][:, self.mesh.free_nodes]
        self.groups = hessian_tools.color(Hstr)

        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, self.groups, dfun1)
        _ = ddfun1(np.array(self.u_jax, dtype=np.float64))

        return fun1, dfun1, hessian_tools.HessGeneratorSparse(ddfun1, solver=solver, verbose=verbose, tol=tol,
                                                              maxiter=maxiter)
