import numpy as np

from my_timer import timer_decorator


@timer_decorator
def newton(f, df, ddf, x0, tolf=1e-6, tolg=1e-3, maxit=1000):
    """
    Newton's method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    df : function
        The gradient of the objective function.
    ddf : function
        The Hessian of the objective function.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    x = x0
    fx = f(x)
    g = df(x)
    H = ddf(x)

    it = 0
    while np.linalg.norm(g) > tolg:
        # Newton's step
        h = -np.linalg.solve(H, g)

        # Update x and function value
        x = x + h
        fxn = f(x)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        fx = fxn
        g = df(x)
        H = ddf(x)

        it += 1
        print(f"it={it}, f={fx}, ||g||={np.linalg.norm(g)}")

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    return x, it


@timer_decorator
def trust_region(f, df, ddf, x0, c0=1.0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
    """
    Trust Region (quasi-Newton method)

    Parameters
    ----------
    fun : function
        The objective function to be minimized.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    c0 : float
        The initial trust region size.
    tol : float
        The tolerance for the stopping condition.

    Returns
    -------
    xmin : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    c = c0
    x = x0
    fx = f(x)
    g = df(x)
    H = ddf(x)

    it = 0
    while np.linalg.norm(g) > tolg:
        # Trial step
        h = -np.linalg.solve(H + c * np.eye(len(x)), g)
        # Quadratic model of function f
        m = fx + np.dot(g.T, h) + 0.5 * np.dot(np.dot(h.T, H), h)
        fxn = f(x + h)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        rho = (fx - fxn) / (fx - m)

        if rho >= 0.1:
            xn = x + h
            g = df(xn)
            H = ddf(xn)
            it += 1
            if verbose:
                print(f"it={it}, f={fx}, c={c}, ||g||={np.linalg.norm(g)}")
        else:
            xn = x
            fxn = fx

        # Adjust the size of the trust region
        if rho > 0.75:
            c *= 0.5
        elif rho < 0.1:
            c *= 2

        x = xn
        fx = fxn

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    print(f"it={it}, f={fx}, c={c}, ||g||={np.linalg.norm(g)}")
    return x, it


@timer_decorator
def newton_bfgs(f, df, x0, tolf=1e-6, tolg=1e-3, maxit=1000):
    """
    BFGS quasi-Newton method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    df : function
        The gradient of the objective function.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    x = x0
    fx = f(x)
    g = df(x)
    H = np.eye(len(x))  # Start with the identity matrix

    it = 0
    while np.linalg.norm(g) > tolg:
        # BFGS step
        h = -np.linalg.solve(H, g)

        # Update x and function value
        s = h
        x = x + h
        fxn = f(x)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        y = df(x) - g
        # BFGS update
        H = H - np.outer(H @ s, s @ H) / (s @ H @ s) + np.outer(y, y) / (y @ s)

        fx = fxn
        g = df(x)

        it += 1
        print(f"it={it}, f={fx}, ||g||={np.linalg.norm(g)}")

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    return x, it


@timer_decorator
def trust_region_bfgs(f, df, x0, c0=1.0, tolf=1e-6, tolg=1e-3, maxit=1000):
    """
    Trust Region (BFGS quasi-Newton method)

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    c0 : float
        The initial trust region size.
    tol : float
        The tolerance for the stopping condition.

    Returns
    -------
    xmin : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    c = c0
    x = x0
    fx = f(x)
    g = df(x)
    H = np.eye(len(x))  # Start with the identity matrix

    it = 0
    while np.linalg.norm(g) > tolg:
        # Trial step
        h = -np.linalg.solve(H + c * np.eye(len(x)), g)
        # Quadratic model of function f
        m = fx + np.dot(g.T, h) + 0.5 * np.dot(np.dot(h.T, H), h)
        fxn = f(x + h)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        rho = (fx - fxn) / (fx - m)

        if not rho < 0:
            s = h
            y = df(x + h) - g
            # BFGS update
            H = H - np.outer(H @ s, s @ H) / (s @ H @ s) + np.outer(y, y) / (y @ s)
            x = x + h
            g = df(x)
        else:
            fxn = fx

        # Adjust the size of the trust region
        if rho > 0.75:
            c *= 0.5
        elif rho < 0.1:
            c *= 2

        fx = fxn

        it += 1
        print(f"it={it}, f={fx}, c={c}, rho={rho}, ||g||={np.linalg.norm(g)}")

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    return x, it


@timer_decorator
def quasi_newton(f, x0, tolf=1e-6, tolg=1e-3, h_diff=1e-3, maxit=1000):
    """
    Quasi-Newton method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    h_diff : float
        The difference used in the finite difference approximation for the gradient.
    maxit : int
        The maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    def df(xx):
        x = np.array(xx)
        n = len(x)
        df_array = np.zeros(n)
        fx = f(x)
        for i in range(n):
            x[i] += h_diff
            df_array[i] = (f(x) - fx) / h_diff
            x[i] -= h_diff
        return df_array

    x = x0
    fx = f(x)
    g = df(x)
    H = np.eye(len(x))  # Start with the identity matrix

    it = 0
    while np.linalg.norm(g) > tolg:
        # Quasi-Newton step
        h = -np.linalg.solve(H, g)

        # Update x and function value
        s = h
        x = x + h
        fxn = f(x)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        y = df(x) - g
        # BFGS update
        H = H - np.outer(H @ s, s @ H) / (s @ H @ s) + np.outer(y, y) / (y @ s)

        fx = fxn
        g = df(x)

        it += 1
        print(f"it={it}, f={fx}, ||g||={np.linalg.norm(g)}")

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    return x, it


@timer_decorator
def trust_region_quasi(f, x0, c0=1.0, tolf=1e-6, tolg=1e-3, h_diff=1e-3, maxit=1000, verbose=False):
    """
    Trust Region (BFGS quasi-Newton method)

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    c0 : float
        The initial trust region size.
    tol : float
        The tolerance for the stopping condition.

    Returns
    -------
    xmin : numpy.ndarray
        The found minimum.
    it : int
        The number of iterations.
    """

    def df(xx):
        x = np.array(xx)
        n = len(x)
        df_array = np.zeros(n)
        fx = f(x)
        for i in range(n):
            x[i] += h_diff
            df_array[i] = (f(x) - fx) / h_diff
            x[i] -= h_diff
        return df_array

    c = c0
    x = x0
    fx = f(x)
    g = df(x)
    H = np.eye(len(x))  # Start with the identity matrix

    it = 0
    while np.linalg.norm(g) > tolg:
        # Trial step
        h = -np.linalg.solve(H + c * np.eye(len(x)), g)
        # Quadratic model of function f
        m = fx + np.dot(g.T, h) + 0.5 * np.dot(np.dot(h.T, H), h)
        fxn = f(x + h)

        # check stopping condition for f
        if np.abs(fx - fxn) < tolf:
            print("Stopping condition for f is satisfied")
            break

        rho = (fx - fxn) / (fx - m)

        if not rho < 0:
            s = h
            y = df(x + h) - g
            # BFGS update
            H = H - np.outer(H @ s, s @ H) / (s @ H @ s) + np.outer(y, y) / (y @ s)
            x = x + h
            g = df(x)
        else:
            fxn = fx

        # Adjust the size of the trust region
        if rho > 0.75:
            c *= 0.5
        elif rho < 0.1:
            c *= 2

        fx = fxn

        it += 1
        if verbose:
            print(f"it={it}, f={fx}, c={c}, rho={rho}, ||g||={np.linalg.norm(g)}")

        if it > maxit:
            print("Maximum number of iterations reached")
            break
    else:
        print("Stopping condition for g is satisfied")

    print(f"it={it}, f={fx}, c={c}, rho={rho}, ||g||={np.linalg.norm(g)}")
    return x, it


@timer_decorator
def adam(f, df, x0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, maxit=1000):
    """
    Metoda Adam pro minimalizaci funkce

    Parametry
    ----------
    f : funkce
        Optimalizovaná funkce.
    df : funkce
        Gradient optimalizované funkce.
    x0 : numpy.ndarray
        Počáteční odhad minima.
    alpha : float
        Rychlost učení.
    beta1 : float
        Parametr pro odhad momentu prvního řádu.
    beta2 : float
        Parametr pro odhad momentu druhého řádu.
    epsilon : float
        Malá hodnota k zabránění dělení nulou.
    maxit : int
        Maximální počet iterací.

    Návratové hodnoty
    -------
    x : numpy.ndarray
        Nalezené minimum.
    it : int
        Počet iterací.
    """
    x = x0
    m = 0
    v = 0

    for it in range(1, maxit + 1):
        g = df(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**it)
        v_hat = v / (1 - beta2**it)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        print(f"it={it}, f={f(x)}, ||g||={np.linalg.norm(g)}")
        if np.linalg.norm(g) < epsilon:
            print("Podmínka pro ukončení podle hodnot gradientu byla splněna")
            break

    return x
