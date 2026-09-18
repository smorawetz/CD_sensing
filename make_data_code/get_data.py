import numpy as np

from time_evol import time_evolve_adiabatic, time_evolve_univ
from helper_funcs import comp_AGP_coeffs

Deltas = np.loadtxt("universal_data.txt")[:, 1]


def evol_wf_univ(S, chi, g, h, tau, order, Omega, Nsteps, lam_func, dlam_func):
    # compute lower end of window using optimal
    Delta = Deltas[order - 1] * Omega
    coeffs = comp_AGP_coeffs(Delta, Omega, order)

    # get time-evolved state
    H_params = (chi, g, h)
    _, psi_t = time_evolve_univ(
        S, H_params, coeffs, order, tau, Nsteps, lam_func, dlam_func
    )
    return psi_t[:, -1]


def evol_wf_noCD(S, chi, g, h, tau, Nsteps, lam_func):
    # get time-evolved state
    H_params = (chi, g, h)
    _, psi_t = time_evolve_adiabatic(S, H_params, tau, Nsteps, lam_func)
    return psi_t[:, -1]
