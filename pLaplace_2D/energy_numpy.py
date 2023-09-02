from pLaplace_2D.mesh import Mesh2D
from pLaplace_2D.suplement_matrices import mass_matrix, incidence_matrix
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import hessian_tools  # noqa


def _energy_numpy(u, u_0, free_nodes, elements, gradx, grady, p, f, areas):
    v = u_0.copy()
    v[free_nodes] = u

    v_elems = v[elements]
    v_x_elems = np.sum(gradx * v_elems, axis=1) ** 2
    v_y_elems = np.sum(grady * v_elems, axis=1) ** 2
    intgrds1 = (1 / p) * (v_x_elems + v_y_elems) ** (p/2)
    e = np.sum(areas * intgrds1) - np.dot(f, v)
    return e


def _grad_energy_numpy(u, u_0, free_nodes, elements, gradx, grady, p, f, areas):
    v = u_0.copy()
    v[free_nodes] = u

    v_elems = v[elements]
    tmpx = gradx * v_elems
    tmpy = grady * v_elems
    tmpsumx = np.sum(tmpx, axis=1)
    tmpsumy = np.sum(tmpy, axis=1)
    v_x_elems = tmpsumx ** 2
    v_y_elems = tmpsumy ** 2
    intgrds1 = areas * (v_x_elems + v_y_elems) ** (p/2 - 1)

    res = intgrds1[:, None] * (tmpsumx[:, None] * gradx + tmpsumy[:, None] * grady)

    grad = np.zeros_like(v) - f
    np.add.at(grad, elements[:], res[:])

    return grad[free_nodes]


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

        self.free_nodes = self.mesh.free_nodes
        self.elements = self.mesh.elements
        self.gradx = self.mesh.gradx
        self.grady = self.mesh.grady
        self.areas = self.mesh.areas
        self.u = np.zeros_like(self.free_nodes, dtype=np.float64)

    def _only_grad_val(self):
        # automatická derivace a kompilace
        def fun1(v_internal): return _energy_numpy(v_internal, self.u_0, self.free_nodes, self.elements,
                                                   self.gradx, self.grady, self.p, self.fx, self.areas)

        def dfun1(v_internal): return _grad_energy_numpy(v_internal, self.u_0, self.free_nodes, self.elements,
                                                         self.gradx, self.grady, self.p, self.fx, self.areas)
        return fun1, dfun1

    def recompile(self, solver="cg", verbose=False, tol=1e-3, maxiter=5000):
        fun1, dfun1 = self._only_grad_val()
        return fun1, dfun1, None


class EnergySFD(Energy):
    def recompile(self, solver="cg", verbose=False, tol=1e-3, maxiter=5000):
        # automatická derivace a kompilace
        fun1, dfun1 = self._only_grad_val()
        # create three diagonal matrix

        Hstr = incidence_matrix(self.mesh)[self.mesh.free_nodes, :][:, self.mesh.free_nodes]
        self.groups = hessian_tools.color(Hstr)

        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, self.groups, dfun1)

        return fun1, dfun1, hessian_tools.HessGeneratorSparse(ddfun1, solver=solver, verbose=verbose, tol=tol,
                                                              maxiter=maxiter)
