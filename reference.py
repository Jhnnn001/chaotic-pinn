"""Tolerance-checked reference, finite-time Lyapunov estimate, and RK4."""
import numpy as np
from scipy.integrate import solve_ivp

from dynamics import T_END, rhs

RTOL = ATOL = 1e-12


def integrate(x0, t_end=T_END, rtol=RTOL, atol=ATOL):
    """Dense DOP853 solution; `.sol(t)` has shape (4, len(t))."""
    x0 = np.asarray(x0, float)
    if x0.shape != (4,) or not np.isfinite(x0).all() or not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("x0 must be four finite values and t_end must be positive")
    solution = solve_ivp(
        lambda t, x: rhs(x), (0.0, t_end), np.asarray(x0, float),
        method="DOP853", rtol=rtol, atol=atol, dense_output=True,
    )
    if not solution.success or solution.t[-1] < t_end:
        raise RuntimeError(f"Reference integration failed: {solution.message}")
    return solution


def trust_horizon(x0, t, tol=1e-6):
    """First excluded sample, or None if both tolerances agree over all t.

    This is a numerical consistency check, not a rigorous global error bound.
    """
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all() or t[0] < 0 or t[-1] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be a finite, strictly increasing nonnegative grid")
    a = integrate(x0, t_end=t[-1]).sol(t)
    b = integrate(x0, t_end=t[-1], rtol=RTOL / 10, atol=ATOL / 10).sol(t)
    sep = np.abs(a[:2] - b[:2]).max(axis=0)
    idx = np.argmax(sep > tol)
    return float(t[idx]) if sep[idx] > tol else None


def lyapunov(x0, n_renorm=400, tau=0.25, delta0=1e-8, rtol=1e-10):
    """Finite-time largest Lyapunov estimate: integrate a
    trajectory and a delta0-displaced copy together, renormalize the separation
    every `tau`, average log growth over n_renorm * tau seconds."""

    def both(t, s):
        return np.concatenate([rhs(s[:4]), rhs(s[4:])])

    s = np.concatenate([x0, np.asarray(x0, float) + [delta0, 0, 0, 0]])
    total = 0.0
    for _ in range(n_renorm):
        solution = solve_ivp(both, (0.0, tau), s, method="DOP853", rtol=rtol, atol=rtol)
        if not solution.success:
            raise RuntimeError(f"Lyapunov integration failed: {solution.message}")
        s = solution.y[:, -1]
        d = np.linalg.norm(s[4:] - s[:4])
        if not np.isfinite(d) or d == 0:
            raise RuntimeError("Lyapunov separation is zero or nonfinite")
        total += np.log(d / delta0)
        s[4:] = s[:4] + (s[4:] - s[:4]) * (delta0 / d)
    return total / (n_renorm * tau)


def rk4(x0, t):
    """Classical fixed-step RK4 on the grid t; returns (len(t), 4)."""
    x = np.empty((len(t), 4))
    x[0] = x0
    for i in range(len(t) - 1):
        h, s = t[i + 1] - t[i], x[i]
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h * k2)
        k4 = rhs(s + h * k3)
        x[i + 1] = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
