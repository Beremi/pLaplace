import scipy.sparse as sp
from jax.experimental import sparse

import jax
from jax import config
import jax.numpy as jnp
from HyperElasticity_3D.mesh import MeshHyperelasticity3D as Mesh
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import hessian_tools  # noqa
from tools import sparsejac  # noqa
config.update("jax_enable_x64", True)


def energy_jax(u, u0, dofsMinim, elems2nodes, dphix, dphiy, dphiz, vol, C1, D1):
    v = jnp.array(u0, dtype=jnp.float64)
    v = v.at[dofsMinim].set(u)
    vx = v[0::3][elems2nodes]
    vy = v[1::3][elems2nodes]
    vz = v[2::3][elems2nodes]

    G11 = jnp.sum(vx * dphix, axis=1)
    G12 = jnp.sum(vx * dphiy, axis=1)
    G13 = jnp.sum(vx * dphiz, axis=1)
    G21 = jnp.sum(vy * dphix, axis=1)
    G22 = jnp.sum(vy * dphiy, axis=1)
    G23 = jnp.sum(vy * dphiz, axis=1)
    G31 = jnp.sum(vz * dphix, axis=1)
    G32 = jnp.sum(vz * dphiy, axis=1)
    G33 = jnp.sum(vz * dphiz, axis=1)

    I1 = G11**2 + G12**2 + G13**2 + G21**2 + G22**2 + G23**2 + G31**2 + G32**2 + G33**2
    det = G11 * G22 * G33 - G11 * G23 * G32 - G12 * G21 * G33 + G12 * G23 * G31 + G13 * G21 * G32 - G13 * G22 * G31
    W = C1 * (I1 - 3 - 2 * jnp.log(det)) + D1 * (det - 1)**2
    return jnp.sum(W * vol)


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
    def __init__(self, mesh: Mesh, alpha_step: float = 1.0):
        self.alpha = 0.0
        self.all_alphas = [self.alpha]
        self.alpha_step = alpha_step
        self.u0_previous = jnp.array(mesh.nodes2coord, dtype=jnp.float64).ravel()
        self.mesh = mesh
        self.change_problem(mesh)

    def apply_rotation(self, alpha_step: float = 0):
        if alpha_step != 0:
            self.alpha_step = alpha_step
            
        self.alpha += self.alpha_step
        self.all_alphas.append(self.alpha)
        u0 = jnp.array(data.nodes2coord, dtype=jnp.float64).ravel()
        nodes = np.where(data.nodes2coord[:, 0] == data.lx)[0]
        u0 = u0.at[nodes * 3 + 1].set(np.cos(alpha) * data.nodes2coord[nodes, 1] +
                                      np.sin(alpha) * data.nodes2coord[nodes, 2])
        u0 = u0.at[nodes * 3 + 2].set(-np.sin(alpha) * data.nodes2coord[nodes, 1] +
                                      np.cos(alpha) * data.nodes2coord[nodes, 2])
        
        
    def change_problem(self, mesh: Mesh | None = None):
        alpha_current = rotation * self.alpha
        self.all_alphas.append(alpha_current)
        if mesh is None:
            pass

        u0 = jnp.array(data.nodes2coord, dtype=jnp.float64).ravel()
        nodes = np.where(data.nodes2coord[:, 0] == data.lx)[0]
        u0 = u0.at[nodes * 3 + 1].set(np.cos(alpha) * data.nodes2coord[nodes, 1] +
                                      np.sin(alpha) * data.nodes2coord[nodes, 2])
        u0 = u0.at[nodes * 3 + 2].set(-np.sin(alpha) * data.nodes2coord[nodes, 1] +
                                      np.cos(alpha) * data.nodes2coord[nodes, 2])

        # self.p = p
        # self.mesh = mesh

        # self.M = mass_matrix(self.mesh)
        # self.fx = self.M @ f
        # self.u_0 = np.zeros_like(self.fx)

        # self.u_0_jax = jnp.array(self.u_0, dtype=jnp.float64)
        # self.free_nodes_jax = jnp.array(self.mesh.free_nodes, dtype=int)
        # self.elements_jax = jnp.array(self.mesh.elements, dtype=int)
        # self.gradx_jax = jnp.array(self.mesh.gradx, dtype=jnp.float64)
        # self.grady_jax = jnp.array(self.mesh.grady, dtype=jnp.float64)

        # self.fx_jax = jnp.array(self.fx, dtype=jnp.float64).ravel()
        # self.areas_jax = jnp.array(self.mesh.areas, dtype=jnp.float64).ravel()
        # self.u_jax = jnp.zeros_like(self.free_nodes_jax, dtype=jnp.float64)
        # self.u = np.zeros_like(self.free_nodes_jax, dtype=np.float64)

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
