import numpy as np

from .suplement_sparse_matrices import laplace_1D, sparse_tridiagonal_scipy, ff_mass_construct

import os
import sys
# add .. to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import hessian_tools  # noqa


def _energy_numpy(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * np.abs(vx)**p - fx * v_mid
    return np.sum(h * Jv_density)


def _energy_numpy_grad(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    tmp1 = np.sign(v[1:-1] - v[:-2]) * (np.abs(v[1:-1] - v[:-2]) / h[:-1]) ** (p - 1)
    tmp2 = np.sign(v[2:] - v[1:-1]) * (np.abs(v[2:] - v[1:-1]) / h[1:]) ** (p - 1)
    return tmp1 - tmp2 - fx[1:-1]


# rhs
def f_default(x):
    return -10 * np.ones(x.size)


class Energy:
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

    def _only_grad_val(self):
        def fun1(v_internal): return _energy_numpy(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numpy_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        return fun1, dfun1

    def recompile(self):
        fun1, dfun1 = self._only_grad_val()
        return fun1, dfun1, None


class EnergySFD(Energy):
    def recompile(self):
        fun1, dfun1 = self._only_grad_val()
        Hstr = sparse_tridiagonal_scipy(self.ne - 1)
        groups = hessian_tools.color(Hstr)
        def ddfun1(v_internal): return hessian_tools.sfd(v_internal, dfun1(v_internal), Hstr, groups, dfun1)
        return fun1, dfun1, hessian_tools.HessGeneratorSparse(ddfun1)


class EnergyLaplace(Energy):
    def recompile(self):
        fun1, dfun1 = self._only_grad_val()
        H = laplace_1D(self.x)
        return fun1, dfun1, hessian_tools.HessGeneratorLaplace(H)
