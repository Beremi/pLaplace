# %%
import matplotlib.pyplot as plt
import scipy.optimize as spo
import jax.numpy as np
jnp = np
import jax

from my_timer import timer_decorator
from my_solvers import trust_region

# %% [markdown]
# # Definice funkcí s jax knihovnou pro možnost automatické derivace (np je nyní z jax.numpy)

# %%
def energy_jax(v_internal, fx, v, h, p):
    v = v.at[1:-1].set(v_internal)
    vx = (v[1:] - v[:-1]) / h
    v_mid = (v[1:] + v[:-1]) / 2
    Jv_density = (1 / p) * np.abs(vx)**p - fx * v_mid

    return np.sum(h * Jv_density)


# exact minimizer for p=2
def u_init(x):
    return 0 * (x + 1) * (x - 1)


# rhs
def f(x):
    return -10 * np.ones(x.size)


# %% [markdown]
# # úloha stejně jako v pLaPlace_numba

# %%
p, a, b = 3, -1, 1
ne = 1000
x = np.linspace(a, b, ne + 1)
h = np.diff(x)

v = u_init(x)            # testing function
v_internal = v[1:-1].copy()
x0 = v_internal.copy()   # initial guess


x_mid = (x[1:] + x[:-1]) / 2
fx = f(x_mid)

# %% [markdown]
# # Definování funkce, gradientu a hesiánu; nastavení automatické kompilace jit; vyrobení funce s jedním vstupem

# %%
# automatická derivace a kompilace
fun = jax.jit(energy_jax)
dfun = jax.jit(jax.grad(energy_jax, argnums=0))
ddfun = jax.jit(jax.hessian(energy_jax, argnums=0))
fun1 = lambda v_internal: fun(v_internal, fx, v, h, p)
dfun1 = lambda v_internal: dfun(v_internal, fx, v, h, p)
ddfun1 = lambda v_internal: ddfun(v_internal, fx, v, h, p)

# %% [markdown]
# # V následujících třech buňkách se s prvním zavoláním funkce rovnou kompiluje jit 

# %%
print(f"Initial energy: {fun1(v_internal)}")

# %%
print(f"||g||={np.linalg.norm(dfun1(v_internal))}")

# %%
print(f"||H||={np.linalg.norm(ddfun1(v_internal))}")

# %% [markdown]
# # vyzkoušení řešení úlohy pomocí trust regionu z my_solvers.py
# u mně to trvalo cca 0.5s pro úlohu s dělením 200 a p=3
# 
# pro dělení 500 a p=3 to trvalo cca 2.1s v porovnání s implementací bez derivací s numbou 27s

# %%
solopt, iterations = trust_region(fun1 , dfun1, ddfun1, x0, c0=1, tolf=1e-6, tolg=1e-6, maxit=1000, verbose = False)

# %% [markdown]
# # Pokus o vyřešení pomocí scipy.optimize.minimize, zde bez použití gradientu a hesiánu, pozor vrací špatný výsledek

# %%
# comparison with scipy minimization
from scipy.optimize import minimize


minimize_timed = timer_decorator(minimize)

print("energy (init)=", fun(v_internal, fx, v, h, p))


solopt = minimize_timed(fun, v_internal, args=(fx, v, h, p))

print("energy (final)=", fun(solopt.x, fx, v, h, p))

# %% [markdown]
# # Verze trust regionu v scipy.optimize.minimize, správný výsledek, ale trvá déle než předchozí trust region (vevnitř se zřejmě děje něco jinak (víc iterací))

# %%
print("energy (init)=", fun(v_internal, fx, v, h, p))


solopt = minimize_timed(fun, v_internal, args=(fx, v, h, p), method='trust-constr', jac=dfun, hess=ddfun)

print("energy (final)=", fun(solopt.x, fx, v, h, p))
solopt.nit

# %% [markdown]
# # větší úloha

# %%
p, a, b = 3, -1, 1
ne = 1000
x = np.linspace(a, b, ne + 1)
h = np.diff(x)

v = u_init(x)            # testing function
v_internal = v[1:-1].copy()
x0 = v_internal.copy()   # initial guess


x_mid = (x[1:] + x[:-1]) / 2
fx = f(x_mid)

# %%
solopt, iterations = trust_region(fun1 , dfun1, ddfun1, x0, c0=1, tolf=1e-6, tolg=1e-6, maxit=1000, verbose = False)

# %% [markdown]
# ### scipy je třeba nastavit mimo default, jinak nedořeší v 1000 iteracích

# %%
solopt = minimize_timed(fun1, v_internal, method='trust-constr', jac=dfun1, hess=ddfun1)

print("energy (final)=", fun(solopt.x, fx, v, h, p))
solopt.nit

# %% [markdown]
# # profilování trust regionu
# - řeší s hustými maticemi (nevím jestli jax zvládne sparse)
# - 171 iterací vs v článku bylo 37




