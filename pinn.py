"""Time-marching PINN for the double pendulum.

The horizon is split into windows [t_k, t_k + W]. In each window a fresh Tanh
MLP represents the increment from the window's initial state:
    x(t) = x_k + tau * net(tau),   tau = (t - t_k) / W in [0, 1]
so the initial condition holds exactly (hard constraint, no IC loss term).
The physics residual dx/dt - f(x) is formed with autograd at collocation
points resampled every Adam step, then polished with L-BFGS on a fixed grid.
An energy term penalizes deviations from the initial energy E0. The end state
of window k becomes x_{k+1}; failed windows are retried at half the width and
raise an error if the minimum width still cannot meet the residual tolerance.
"""
import numpy as np
import torch

from dynamics import energy, rhs

torch.set_default_dtype(torch.float64)


def mlp(width=64, depth=3):
    layers, n_in = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
        n_in = width
    layers.append(torch.nn.Linear(n_in, 4))
    return torch.nn.Sequential(*layers)


class Window:
    def __init__(self, x_k, t_k, width_t, e0, seed, width=64, depth=3):
        torch.manual_seed(seed)  # a fresh network per window
        self.net = mlp(width, depth)
        self.x_k = torch.as_tensor(np.asarray(x_k, float))
        self.t_k, self.W, self.e0 = t_k, width_t, e0

    def state(self, tau):  # tau: (N, 1) in [0, 1]
        return self.x_k + tau * self.net(tau)

    def loss(self, tau, w_energy):
        tau = tau.requires_grad_(True)
        x = self.state(tau)
        dxdt = torch.stack(
            [torch.autograd.grad(x[:, i].sum(), tau, create_graph=True)[0][:, 0] for i in range(4)], -1
        ) / self.W
        residual = ((dxdt - rhs(x, torch)) ** 2).mean()  # components are unweighted
        drift = ((energy(x, torch) - self.e0) ** 2).mean()
        return residual + w_energy * drift, residual, drift


def train_window(win, n_adam, n_lbfgs, n_points, w_energy, lr):
    """Adam on resampled points, then L-BFGS on a fixed grid (needs a
    deterministic objective). Returns final (residual, drift)."""
    opt = torch.optim.Adam(win.net.parameters(), lr)
    for _ in range(n_adam):
        opt.zero_grad()
        total, _, _ = win.loss(torch.rand(n_points, 1), w_energy)
        total.backward()
        opt.step()

    tau = torch.linspace(0.0, 1.0, n_points).view(-1, 1)
    opt = torch.optim.LBFGS(
        win.net.parameters(), max_iter=n_lbfgs, history_size=50,
        line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        total, _, _ = win.loss(tau.clone(), w_energy)
        total.backward()
        return total

    opt.step(closure)
    _, residual, drift = win.loss(tau.clone(), w_energy)
    return residual.item(), drift.item()


def solve(x0, t_end, window, seed=0, residual_tol=1e-3, min_window=None, width=64, depth=3, **train_kw):
    """March windows over [0, t_end]. A window whose final residual exceeds
    residual_tol is retrained with half the width (step-size control, down to
    min_window = window / 8). Returns (evaluate, log): evaluate(t) gives the
    state on any grid as (len(t), 4); log has one dict per accepted window."""
    x0 = np.asarray(x0, float)
    if x0.shape != (4,) or not np.isfinite(x0).all():
        raise ValueError("x0 must contain four finite state values")
    if not np.isfinite([t_end, window, residual_tol]).all() or min(t_end, window, residual_tol) <= 0:
        raise ValueError("t_end, window, and residual_tol must be positive and finite")
    e0 = float(energy(x0))
    min_window = window / 8 if min_window is None else min_window
    if not np.isfinite(min_window) or not 0 < min_window <= window:
        raise ValueError("min_window must be positive and no larger than window")
    x_k, t_k, W, windows, log, attempt = x0, 0.0, window, [], [], 0
    while t_k < t_end:
        W = min(W, t_end - t_k)
        win = Window(x_k, t_k, W, e0, seed + attempt, width, depth)
        attempt += 1
        residual, drift = train_window(win, **train_kw)
        if not np.isfinite([residual, drift]).all():
            raise RuntimeError(f"Nonfinite training loss at t={t_k:g} s")
        if residual > residual_tol:
            if W / 2 >= min_window:
                W /= 2
                continue
            raise RuntimeError(f"Window at t={t_k:g} s failed: residual {residual:g} > {residual_tol:g}")
        log.append(dict(t_k=t_k, W=W, residual=residual, drift=drift))
        windows.append(win)
        with torch.no_grad():
            x_k = win.state(torch.ones(1, 1))[0].numpy()
        t_k += W
        W = window  # back to the nominal width after every window

    def evaluate(t):
        t = np.asarray(t, float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > t_end)):
            raise ValueError(f"Evaluation times must be a finite 1D grid in [0, {t_end:g}]")
        out = np.empty((len(t), 4))
        with torch.no_grad():
            for win in windows:
                m = (t >= win.t_k) & (t <= win.t_k + win.W + 1e-9)
                tau = torch.as_tensor((t[m] - win.t_k) / win.W).view(-1, 1)
                out[m] = win.state(tau).numpy()
        return out

    return evaluate, log
