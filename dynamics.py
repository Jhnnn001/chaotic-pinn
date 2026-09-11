"""Double pendulum: equations of motion shared by the numerical reference and the PINN.

State x = (theta1, theta2, omega1, omega2) on the last axis. Every function takes
`lib` = numpy or torch so the same code serves scipy's integrator and autograd.
Equations derived from the Lagrangian in mass-matrix form
    M(theta) omega_dot = r(theta, omega)
and solved in closed form for the 2x2 system. The masses are points and the
rods are massless; angles are measured from the downward vertical.
"""
import math

import numpy as np

M1 = M2 = 1.0
L1 = L2 = 1.0
G = 9.8
T_END = 10.0
# Potential-energy amplitude, 29.4 J; the full range is 58.8 J.
E_SCALE = (M1 + M2) * G * L1 + M2 * G * L2

# name -> (theta1, theta2, omega1, omega2) at t = 0, all released from rest.
INITIAL_CONDITIONS = {
    "pi3_0": (math.pi / 3, 0.0, 0.0, 0.0),
    "2pi3_2pi3": (2 * math.pi / 3, 2 * math.pi / 3, 0.0, 0.0),
}


def rhs(x, lib=np):
    th1, th2, w1, w2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    d = th1 - th2
    cd, sd = lib.cos(d), lib.sin(d)
    m11 = (M1 + M2) * L1**2
    m12 = M2 * L1 * L2 * cd
    m22 = M2 * L2**2
    r1 = -M2 * L1 * L2 * w2**2 * sd - (M1 + M2) * G * L1 * lib.sin(th1)
    r2 = M2 * L1 * L2 * w1**2 * sd - M2 * G * L2 * lib.sin(th2)
    det = m11 * m22 - m12**2
    a1 = (m22 * r1 - m12 * r2) / det
    a2 = (m11 * r2 - m12 * r1) / det
    return lib.stack([w1, w2, a1, a2], -1)


def energy(x, lib=np):
    th1, th2, w1, w2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    kinetic = (
        0.5 * (M1 + M2) * L1**2 * w1**2
        + 0.5 * M2 * L2**2 * w2**2
        + M2 * L1 * L2 * w1 * w2 * lib.cos(th1 - th2)
    )
    potential = -(M1 + M2) * G * L1 * lib.cos(th1) - M2 * G * L2 * lib.cos(th2)
    return kinetic + potential
