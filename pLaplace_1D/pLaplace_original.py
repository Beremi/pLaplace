import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize


def time_function(f, *args):
    start = time.time()
    res = f(*args)
    end = time.time()
    print(f'{f.__name__}: {(end - start)*1000:.5f} ms')
    return res


def energy(v_internal, x, v, p):

    u_a = v[0]                # v at the left end
    u_b = v[-1]               # v at the right end
    h = x[1] - x[0]             # mesh size

    v = np.concatenate(([u_a], v_internal, [u_b]))
    vx = np.zeros(ne)
    x_mid = np.zeros(ne)
    v_mid = np.zeros(ne)
    Jv_density = np.zeros(ne)
    for i in range(0, ne):
        vx[i] = (v[i + 1] - v[i]) / h
        x_mid[i] = (x[i + 1] + x[i]) / 2
        v_mid[i] = (v[i + 1] + v[i]) / 2
        Jv_density[i] = (1 / p) * np.power(abs(vx[i]), p) - f(x_mid[i]) * v_mid[i]

    return h * np.sum(Jv_density)


# exact minimizer for p=2
def u_init(x):
    return 0 * (x + 1) * (x - 1)

# rhs


def f(x):
    return -10 * np.ones(x.size)


p, a, b = 3, -1, 1
ne = 10
x = np.linspace(a, b, ne + 1)

v = u_init(x)            # testing function
v_internal = v[1:-1]

print("energy (init)=", energy(v_internal, x, v, p))

solopt = minimize(energy, v_internal, args=(x, v, p))
u_internal = solopt.x

print("energy (optim)=", energy(u_internal, x, v, p))

u_a = v[0]                # v at the left end
u_b = v[-1]               # v at the right end
u = np.concatenate(([u_a], u_internal, [u_b]))

plt.plot(x, v)
plt.plot(x, u)
plt.plot(x[[0, -1]], v[[0, -1]], 'ro')
plt.show()
