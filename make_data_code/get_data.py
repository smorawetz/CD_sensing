import numpy as np
import matplotlib.pyplot as plt
import scipy
import quspin

from time_evol import time_evolve_adiabatic, time_evolve_univ
from helper_funcs import (
    cons_ops,
    comp_AGP_coeffs,
    smooth_lam_func,
    dsmooth_lam_func,
    make_fname,
)

## define constants
NTSTEPS = 1000

S = None
chi = None
g = None
h = None
tau = None
order = None
Omega = None

Nsteps = NTSTEPS
lam_func = smooth_lam_func
dlam_func = dsmooth_lam_func

univ_data = np.loadtxt("universal_data.txt")
Deltas = univ_data[:, 1]


def run_save_wf_univ(S, chi, g, h, tau, order, Omega, Nsteps, lam_func, dlam_func):
    # compute lower end of window using optimal
    Delta = Deltas[order - 1] * Omega
    coeffs = comp_AGP_coeffs(Delta, Omega, order)

    # get time-evolved state
    H_params = (chi, g, h)
    _, psi_t = time_evolve_univ(
        S, H_params, coeffs, order, tau, Nsteps, lam_func, dlam_func
    )
    np.savetxt("../data/" + make_fname(S, chi, g, h, tau, order, Omega), psi_t)


run_save_wf_univ(10, 1, 1, 0.01, 1, 3, 3.5, Nsteps, lam_func, dlam_func)
