from numba import jit
import numpy as np

from .suplement_sparse_matrices import laplace_1D, sparse_tridiagonal_scipy, ff_mass_construct
from .energy_numpy import Energy as Energy_numpy

import os
import sys

# add .. to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import hessian_tools  # noqa


@jit(nopython=True)
def _energy_numba(v_internal, f_mid, v, h, p):
    v[1:-1] = v_internal
    n = h.shape[0]
    Jv_density = 0.0
    for i in range(n):
        Jv_density += h[i] * ((1 / p) * np.abs((v[i + 1] - v[i]) / h[i])**p - f_mid[i] * (v[i + 1] + v[i]) / 2)
    return Jv_density


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


class Energy(Energy_numpy):
    def _only_grad_val(self):
        def fun1(v_internal): return _energy_numba(v_internal, self.fx, self.v, self.h, self.p)
        self.f_mass = ff_mass_construct(self.f(self.x), self.ne, self.h)
        def dfun1(v_internal): return _energy_numba_grad(v_internal, self.f_mass, self.v, self.h, self.p)
        _energy_numba.recompile()
        _energy_numba_grad.recompile()
        _ = fun1(self.v_internal)
        _ = dfun1(self.v_internal)
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
