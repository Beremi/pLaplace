import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numba

from my_timer import timer_decorator

def energy_numpy(v_internal, fx, v, h, p):
    v[1:-1] = v_internal
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * np.abs(vx)**p - f_mid * v_mid
    return np.sum(h * Jv_density)

# exact minimizer for p=2
def u_init(x):
    return 0 * (x + 1) * (x - 1)

# rhs
def f(x):
    return -10 * np.ones(x.size)

p, a, b = 3, -1, 1

levels=3
pow=np.linspace(1.,levels,num=levels)
all_ne = np.power(10,pow)

minimize_timed = timer_decorator(minimize)

for ne in all_ne:
    x = np.linspace(a, b, int(ne)+1)
    h = np.diff(x)
    
    v = u_init(x)            # testing function
    v_internal = v[1:-1].copy()
    
    x_mid = (x[1:] + x[:-1]) / 2
    f_mid = f(x_mid)
    
    # print("energy (init)=", energy_numpy(v_internal, f_mid, v, h, p))
    solopt = minimize_timed(energy_numpy, v_internal, args=(f_mid, v, h, p), method='bfgs')
    print("energy (final)=",solopt.fun,", iterations=",solopt.nit)
    print("------")

    



















