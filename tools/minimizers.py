import numpy as np
from scipy.optimize._minimize import minimize
from scipy.optimize._optimize import OptimizeResult


def zlatyrez(f, a, b, x, ddd, tol):
    """
    Find the minimum of a function f using the Golden section method.

    Parameters
    ----------
    f : Callable
        Function to find the minimum of.
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    x : float
        Point to optimize around.
    ddd : int
        Direction of the search.
    tol : float
        Tolerance of the method.

    Returns
    -------
    tuple[float, int]
        Tuple of the argument of minimum and number of iterations.
    """

    # Golden ratio
    gamma = 1 / 2 + np.sqrt(5) / 2

    # Initial values
    a0 = a
    b0 = b
    d0 = (b0 - a0) / gamma + a0
    c0 = a0 + b0 - d0

    # Iteration counter
    it = 0

    # Store the values of the interval and the function
    an = a0
    bn = b0
    cn = c0
    dn = d0
    fcn = f(x + cn * ddd)
    fdn = f(x + dn * ddd)

    while bn - an > tol:
        # Store the values of the interval and the function
        a = an
        b = bn
        c = cn
        d = dn
        fc = fcn
        fd = fdn

        if fc < fd:
            # Update the interval
            an = a
            bn = d
            dn = c
            cn = an + bn - dn

            # Update the function value
            fcn = f(x + cn * ddd)
            fdn = fc
        else:
            # Update the interval
            an = c
            bn = b
            cn = d
            dn = an + bn - cn

            # Update the function value
            fcn = fd
            fdn = f(x + dn * ddd)

        # Increment the iteration counter
        it += 1

    # Return the result
    t = (an + bn) / 2
    return t, it


def newton(f, df, ddf, x0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
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
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    fx = f(x)
    it = 0
    inner_tol = 1e-1
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H = ddf(x)
        normg = np.linalg.norm(g)
        # g = g / normg

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -H.solve(g)
        # h = h / np.linalg.norm(h)

        def g_2D(v):
            a, b = v
            return f(x + a * g + b * h)

        result = minimize(g_2D, [0.0, 0.0], method='Powell', tol=inner_tol)
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * g + b * h
        fxn, fx = fx, f(x)

        if verbose:
            print(f"it={it}, f={fx:.5f}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}")

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) / 10])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def newton2(f, df, ddf, x0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
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
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    fx = f(x)
    it = 0
    inner_tol = 1e-1
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H = ddf(x)
        normg = np.linalg.norm(g)
        # g = g / normg

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -H.solve(g)
        # h = h / np.linalg.norm(h)

        a, nitf = zlatyrez(f, 0, 10, x, h, inner_tol)

        # Update x and function value
        x = x + a * h
        fxn, fx = fx, f(x)

        if verbose:
            print(f"it={it}, f={fx:.5f}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}")

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) / 10])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def newton_multi(f, df, ddf1, ddf2, x0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
    """
    Newton's method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    df : function
        The gradient of the objective function.
    ddf1 : function
        The Hessian of the objective function.
    ddf2 : function
        The Hessian of the objective function.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    fx = f(x)
    it = 0
    inner_tol = 1e-1
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H1 = ddf1(x)
        H2 = ddf2(x)
        normg = np.linalg.norm(g)
        # g = g / normg

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h1 = -H1.solve(g)
        h2 = -H2.solve(g)
        # h = h / np.linalg.norm(h)

        def g_2D(v):
            a, b = v
            return f(x + a * h1 + b * h2)

        result = minimize(g_2D, [0, 0], method='Powell', tol=inner_tol)
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * h1 + b * h2
        fxn, fx = fx, f(x)

        if verbose:
            print(f"it={it}, f={fx:.5f}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}")

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) / 10])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def bfgs(f, df, x0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
    """
    Newton's method for function minimization

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
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    fx = f(x)
    it = 0
    H = np.eye(len(x))  # initial Hessian approximation
    inner_tol = 1e-1
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -np.linalg.solve(H, g)

        def g_2D(v):
            a, b = v
            return f(x + a * g + b * h)

        result = minimize(g_2D, [0, 0], method='Powell', tol=inner_tol)
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * g + b * h
        fxn, fx = fx, f(x)
        y = df(x) - g
        s = a * g + b * h
        # BFGS update
        if (y @ s) > 0:
            H = H - np.outer(H @ s, s @ H) / (s @ H @ s) + np.outer(y, y) / (y @ s)
        else:
            if verbose:
                print("BFGS update failed")
        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) / 10])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

        if verbose:
            print(f"it={it}, f={fx:.5f}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}")

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def trust_region(f, df, ddf, x0, c0=1.0, c_min=0.0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
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
    c0 : float
        The initial trust region size.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    c = c0
    inner_tol = 1e-1
    fx = f(x)
    m = fx
    it = 0
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H = ddf(x)
        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -H.solve_trust(g, c)

        mn, m = m, fx + g @ h + 0.5 * H.norm(h, c)

        def g_2D(v):
            a, b = v
            return f(x + a * g + b * h)

        result = minimize(g_2D, [0, 0], method='Powell', tol=np.max([inner_tol, tolf]))
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * g + b * h
        fxn, fx = fx, f(x)

        rho = (fxn - fx) / np.max([mn - m, fxn - fx])

        if verbose:
            print(f"it={it}, f={fx}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}, c={c:.5e}, rho={rho:.5f}")

        # Adjust the size of the trust region
        if rho > 0.5:
            c = max(c * 0.1, c_min)
        elif rho < 0.1:
            c = max(c * 2, c_min)

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) * 1e-3])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def trust_region2(f, df, ddf, x0, c0=1.0, c_min=0.0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
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
    c0 : float
        The initial trust region size.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    c = c0
    inner_tol = 1e-1
    fx = f(x)
    # m = fx
    it = 0
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H = ddf(x)
        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -H.solve_trust(g, c)

        # mn, m = m, fx + g @ h + 0.5 * H.norm(h, c)

        def g_2D(v):
            a, b = v
            return f(x + a * g + b * h)

        result = minimize(g_2D, [0, 0], method='Powell', tol=np.max([inner_tol, tolf]))
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * g + b * h
        fxn, fx = fx, f(x)

        # rho = (fxn - fx) / np.max([mn - m, fxn - fx])

        if verbose:
            print(f"it={it}, f={fx}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}, c={c:.5e}")

        # Adjust the size of the trust region
        if b > - 2 * a:
            c = max(c * 0.1, c_min)
        elif 2 * b < - a:
            c = max(c * 10, c_min)

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) * 1e-3])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def trust_region3(f, df, ddf, x0, c0=1.0, c_min=0.0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
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
    c0 : float
        The initial trust region size.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    c = c0
    inner_tol = 1e-1
    fx = f(x)
    m = fx
    it = 0
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        H = ddf(x)
        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -H.solve_trust(g, c)

        _, m = m, fx + g @ h + 0.5 * H.norm(h, c)

        a, nitf = zlatyrez(f, 0, 1, x, h, np.max([inner_tol, tolf]))

        # Update x and function value
        x = x + a * h
        fxn, fx = fx, f(x)

        rho = (fxn - fx) / np.max([fxn - m, fxn - fx])

        if verbose:
            print(f"it={it}, f={fx}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, rho={rho:.5e}, c={c:.5e}")

        # Adjust the size of the trust region
        if rho > 0.75:
            c = max(c / 2, c_min)
        elif rho < 0.1:
            c = max(c * 2, c_min)

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) * 1e-1])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res


def bfgs_trust(f, df, x0, c0=1.0, tolf=1e-6, tolg=1e-3, maxit=1000, verbose=False):
    """
    Newton's method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    df : function
        The gradient of the objective function.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    c0 : float
        The initial trust region size.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    x = x0
    c = c0
    inner_tol = 1e-1
    fx = f(x)
    m = fx
    it = 0
    H = np.eye(len(x))  # initial Hessian approximation
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        it += 1

        # Gradient and Hessian
        g = df(x)
        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        # Newton's step
        h = -np.linalg.solve(H + np.eye(len(x)) * c, g)

        mn, m = m, fx + g @ h + 0.5 * h @ (H + np.eye(len(x)) * c) @ h

        def g_2D(v):
            a, b = v
            return f(x + a * g + b * h)

        result = minimize(g_2D, [0, 0], method='Powell', tol=np.max([inner_tol, tolf]))
        nitf = result.nfev
        a, b = result.x

        # Update x and function value
        x = x + a * g + b * h
        fxn, fx = fx, f(x)

        y = df(x) - g
        s = a * g + b * h
        # BFGS update
        if (y @ s) > 0:
            H = H - np.outer(H @ s, s @ H) / \
                (s @ H @ s) + np.outer(y, y) / (y @ s)
        else:
            if verbose:
                print("BFGS update failed")

        rho = (fxn - fx) / np.max([mn - m, fxn - fx])

        if verbose:
            print(f"it={it}, f={fx}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}, b={b:.5e}, c={c:.5e}, rho={rho:.5f}")

        # Adjust the size of the trust region
        if rho > 0.5:
            c *= 0.5
        elif rho < 0.1:
            c *= 2

        # check stopping condition for f
        inner_tol = np.min([inner_tol, np.abs(fx - fxn) * 1e-3])
        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    res = OptimizeResult(x=x, fun=fx, nit=it, message=message)

    return res
