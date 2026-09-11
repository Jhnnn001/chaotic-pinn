"""Run two 10-second PINN trajectories and save results next to this file."""
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinn
from dynamics import E_SCALE, G, INITIAL_CONDITIONS, L1, L2, M1, M2, T_END, energy
from reference import ATOL, RTOL, integrate, lyapunov, rk4, trust_horizon

HERE = Path(__file__).parent
WINDOW = 0.125
NET = dict(width=64, depth=3)
TRAIN = dict(n_adam=2000, n_lbfgs=500, n_points=256, w_energy=1.0, lr=1e-3)
LOST = 0.1


def summarize_errors(t, error, cutoff):
    """Use only samples strictly before the reference cutoff; None means all."""
    trusted = np.ones(len(t), dtype=bool) if cutoff is None else t < cutoff
    if not trusted.any() or not np.isfinite(error[trusted]).all():
        raise ValueError("No finite error values on the trusted reference interval")
    times, values = t[trusted], error[trusted]
    exceed = np.flatnonzero(values > LOST)
    return {
        "last_trusted_sample_s": float(times[-1]),
        "max_angle_error_rad": float(values.max()),
        "angle_error_at_last_trusted_sample_rad": float(values[-1]),
        "tracking_lost_at_s": float(times[exceed[0]]) if len(exceed) else None,
        "tracking_status": "lost" if len(exceed) else (
            "not_lost_within_run" if cutoff is None else "not_lost_before_reference_cutoff"),
    }


def main():
    torch.set_num_threads(1)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    t = np.linspace(0.0, T_END, int(300 * T_END) + 1)
    results = {
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                        "torch_threads": torch.get_num_threads(), "dtype": "float64"},
        "parameters": {"masses_kg": [M1, M2], "lengths_m": [L1, L2], "g_m_per_s2": G,
                       "t_end_s": T_END, "grid_samples": len(t), "rk4_dt_s": float(t[1]),
                       "energy_amplitude_J": E_SCALE, "seed": 0, "window_s": WINDOW,
                       "net": NET, "train": TRAIN, "residual_tolerance": 1e-3,
                       "min_window_s": WINDOW / 8, "reference_rtol": RTOL,
                       "reference_atol": ATOL, "reference_angle_agreement_rad": 1e-6,
                       "tracking_threshold_rad": LOST,
                       "lyapunov_tau_s": 0.25, "lyapunov_delta0": 1e-8,
                       "lyapunov_rtol_and_atol": 1e-10},
        "cases": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    for column, (name, x0) in enumerate(INITIAL_CONDITIONS.items()):
        print(f"Starting {name}: {T_END:g} s, {int(T_END / WINDOW)} nominal windows", flush=True)
        reference = integrate(x0, T_END).sol(t).T
        cutoff = trust_horizon(x0, t)
        estimates = {str(duration): float(lyapunov(x0, n_renorm=int(duration / 0.25)))
                     for duration in (100, 400)}
        start = time.perf_counter()
        classical = rk4(x0, t)
        rk4_wall = time.perf_counter() - start
        start = time.perf_counter()
        evaluate, windows = pinn.solve(x0, T_END, WINDOW, seed=0, **NET, **TRAIN)
        prediction = evaluate(t)
        pinn_wall = time.perf_counter() - start
        ep = np.abs(prediction[:, :2] - reference[:, :2]).max(axis=1)
        er = np.abs(classical[:, :2] - reference[:, :2]).max(axis=1)
        e0 = float(energy(np.asarray(x0)))
        drifts = [np.abs(energy(x) - e0) / E_SCALE for x in (reference, prediction, classical)]
        trusted = np.ones(len(t), dtype=bool) if cutoff is None else t < cutoff
        case = {
            "initial_state": list(x0), "reference_cutoff_s": cutoff,
            "finite_time_lyapunov_per_s": estimates,
            "reference_max_energy_drift_over_amplitude": float(drifts[0].max()),
            "pinn": {**summarize_errors(t, ep, cutoff), "wall_time_s": pinn_wall,
                     "max_energy_drift_over_amplitude": float(drifts[1].max()), "windows": windows},
            "rk4": {**summarize_errors(t, er, cutoff), "wall_time_s": rk4_wall,
                    "max_energy_drift_over_amplitude": float(drifts[2].max())},
        }
        results["cases"][name] = case
        # Untrusted reference-dependent samples remain blank (NaN) in the CSV.
        reference_saved = np.where(trusted[:, None], reference, np.nan)
        csv = np.column_stack([t, trusted.astype(int), reference_saved, prediction, classical,
                               np.where(trusted, ep, np.nan), np.where(trusted, er, np.nan), *drifts])
        state_columns = [f"{method}_{component}" for method in ("reference", "pinn", "rk4")
                         for component in ("theta1_rad", "theta2_rad", "omega1_rad_per_s", "omega2_rad_per_s")]
        header = ["time_s", "reference_trusted", *state_columns, "pinn_angle_error_rad", "rk4_angle_error_rad",
                  "reference_energy_drift_over_amplitude", "pinn_energy_drift_over_amplitude", "rk4_energy_drift_over_amplitude"]
        np.savetxt(output / f"{name}.csv", csv, delimiter=",", header=",".join(header), comments="")
        (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        ax, error_ax = axes[:, column]
        for component, color in [(0, "C0"), (1, "C1")]:
            ax.plot(t, reference_saved[:, component], color=color, label=rf"reference $\theta_{component + 1}$")
            ax.plot(t, prediction[:, component], "--", color=color, label=rf"PINN $\theta_{component + 1}$")
        ax.set_title(name)
        ax.set_ylabel("Angle [rad]")
        ax.legend(fontsize=7, ncol=2)
        error_ax.semilogy(t[trusted], np.maximum(ep[trusted], 1e-16), label="PINN")
        error_ax.semilogy(t[trusted], np.maximum(er[trusted], 1e-16), label="RK4")
        error_ax.axhline(LOST, color="0.5", ls=":", label="0.1 rad tracking threshold")
        if cutoff is not None:
            error_ax.axvline(cutoff, color="red", ls="--", label="reference cutoff")
        error_ax.set(xlabel="Time [s]", ylabel="Maximum angle error [rad]", ylim=(1e-12, 1e2))
        error_ax.legend(fontsize=7)
        print(f"Finished {name}: PINN {pinn_wall:.1f} s, lost={case['pinn']['tracking_lost_at_s']}, "
              f"maximum trusted error={case['pinn']['max_angle_error_rad']:.3g} rad", flush=True)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
