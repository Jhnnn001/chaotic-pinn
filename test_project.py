"""Run with python test_project.py; no test framework required."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import pinn
import reference
import run
from dynamics import E_SCALE, G, INITIAL_CONDITIONS, energy, rhs


def test_physics():
    x = torch.tensor(np.random.default_rng(0).normal(size=(100, 4)), requires_grad=True)
    assert np.allclose(rhs(x.detach().numpy()), rhs(x, torch).detach().numpy())
    de = torch.autograd.grad(energy(x, torch).sum(), x)[0]
    assert (de * rhs(x, torch)).sum(-1).abs().max().item() < 1e-11
    jac = torch.autograd.functional.jacobian(lambda y: rhs(y, torch), torch.zeros(4))
    frequencies = np.sort(np.linalg.eigvals(jac.numpy()).imag)[2:]
    assert np.allclose(frequencies, np.sqrt(G * np.array([2 - np.sqrt(2), 2 + np.sqrt(2)])))


def test_equilibrium_and_initial_condition():
    x0 = INITIAL_CONDITIONS["pi3_0"]
    win = pinn.Window(x0, 0.0, 0.125, energy(np.array(x0)), seed=0)
    assert np.array_equal(win.state(torch.zeros(1, 1)).detach().numpy()[0], x0)
    equilibrium = pinn.Window(np.zeros(4), 0.0, 0.125, -E_SCALE, seed=0)
    for parameter in equilibrium.net.parameters():
        torch.nn.init.zeros_(parameter)
    loss, _, _ = equilibrium.loss(torch.linspace(0, 1, 9).view(-1, 1), 1.0)
    assert loss.item() == 0.0


def test_failed_windows_are_rejected():
    for residual in [1.0, float("nan"), float("inf")]:
        with patch.object(pinn, "train_window", return_value=(residual, 0.0)):
            with np.testing.assert_raises(RuntimeError):
                pinn.solve(INITIAL_CONDITIONS["pi3_0"], 0.125, 0.125, min_window=0.125)


def test_evaluation_domain():
    with patch.object(pinn, "train_window", return_value=(0.0, 0.0)):
        evaluate, _ = pinn.solve(INITIAL_CONDITIONS["pi3_0"], 0.125, 0.125)
    assert evaluate([0.0, 0.125]).shape == (2, 4)
    assert evaluate([]).shape == (0, 4)
    for times in [[-0.01], [0.126], [np.nan], [np.inf], [[0.0]]]:
        with np.testing.assert_raises(ValueError):
            evaluate(times)


def test_reference_endpoint_and_success():
    x0 = INITIAL_CONDITIONS["pi3_0"]
    assert reference.trust_horizon(x0, np.linspace(0.0, 16.0, 1601)) is None
    with patch.object(reference, "solve_ivp", return_value=SimpleNamespace(success=False, message="failed")):
        with np.testing.assert_raises(RuntimeError):
            reference.integrate(x0)
    sol = reference.integrate(x0, t_end=1.0)
    values = sol.sol(np.linspace(0, 1, 101)).T
    assert np.abs(energy(values) - energy(np.array(x0))).max() / E_SCALE < 1e-9


def test_metrics_censor_untrusted_samples():
    assert hasattr(run, "summarize_errors"), "a cutoff-aware metric is required"
    t = np.array([0.0, 1.0, 2.0, 3.0])
    metric = run.summarize_errors(t, np.array([0.0, 0.01, 0.5, 1.0]), 2.0)
    assert metric["tracking_lost_at_s"] is None
    assert metric["tracking_status"] == "not_lost_before_reference_cutoff"
    assert metric["max_angle_error_rad"] == 0.01
    assert metric["last_trusted_sample_s"] == 1.0
    metric = run.summarize_errors(t, np.array([0.0, 0.2, 0.3, 0.4]), None)
    assert metric["tracking_lost_at_s"] == 1.0
    assert metric["tracking_status"] == "lost"


def test_short_trained_window():
    x0 = INITIAL_CONDITIONS["pi3_0"]
    evaluate, log = pinn.solve(x0, 0.2, 0.2, n_adam=500, n_lbfgs=200,
                               n_points=128, w_energy=1.0, lr=1e-3)
    t = np.linspace(0, 0.2, 21)
    error = np.abs(evaluate(t)[:, :2] - reference.integrate(x0, 0.2).sol(t).T[:, :2]).max()
    assert error < 1e-2, error
    assert all(window["residual"] <= 1e-3 for window in log)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
