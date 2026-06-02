import numpy as np
import scipy

from helper_funcs import cons_ops, make_H, get_GS, fit_func

##########################################################
### functions used directly to simulate time evolution ###
##########################################################


def universal_AGP(coeffs, order, H_gen, dH_gen_dlam):
    evals, evecs = np.linalg.eigh(H_gen)

    # write dH_gen_dlam in the eigenbasis of H_gen
    dH_eig = evecs.conj().T @ dH_gen_dlam @ evecs

    # divide by energy differences in eigenbasis, zeros on diagonal
    E_n, E_m = np.meshgrid(evals, evals)
    E_diff = E_n - E_m
    E_diff_poly = fit_func(E_diff, *coeffs)
    A_eig = np.zeros_like(dH_eig, dtype=np.complex128)
    mask = np.abs(E_diff) > 1e-12
    A_eig[mask] = -1j * dH_eig[mask] * E_diff_poly[mask]

    # transform back to original basis
    A_lam = evecs @ A_eig @ evecs.conj().T
    return A_lam


def time_evolve_univ(S, H_params, coeffs, order, tau, Nsteps, lam_func, dlam_func):
    chi, g, h = H_params
    Sx, Sy, Sz = cons_ops(S)
    init_state = get_GS(0, S, H_params, lam_func, (tau,))

    # modify to turn on nonlinear term with \lambda^2
    Sz_sq = Sz @ Sz
    H1 = -chi / (2 * np.sqrt(S * (S + 1))) * Sz_sq
    H2 = -g * Sx
    H3 = -h * Sz

    tvals = np.linspace(0, tau, Nsteps)
    dt = tvals[1] - tvals[0]
    y_out = np.zeros((len(init_state), Nsteps), dtype=np.complex128)

    psi = init_state
    y_out[:, 0] = psi

    for i in range(1, Nsteps):
        t_mid = tvals[i - 1] + dt / 2
        lam = lam_func(t_mid, tau)
        dlam_dt = dlam_func(t_mid, tau)

        H = lam * (H1 + H3) + (1 - lam) * H2
        dH = H1 + H3 - H2

        # H = lam**2 * H3 + lam * H1 + (1 - lam) * H2
        # dH = 2 * lam * H3 + H1 - H2

        A_lam = universal_AGP(coeffs, order, H, dH)

        Hcd = H + dlam_dt * A_lam
        U = scipy.linalg.expm(-1j * dt * Hcd)

        psi = U @ psi
        y_out[:, i] = psi

    return tvals, y_out


def time_evolve_adiabatic(S, H_params, tau, Nsteps, lam_func):
    chi, g, h = H_params
    Sx, Sy, Sz = cons_ops(S)
    init_state = get_GS(0, S, H_params, lam_func, (tau,))

    # modify to turn on nonlinear term with \lambda^2
    Sz_sq = Sz @ Sz
    H1 = -chi / (2 * np.sqrt(S * (S + 1))) * Sz_sq
    H2 = -g * Sx
    H3 = -h * Sz

    tvals = np.linspace(0, tau, Nsteps)
    dt = tvals[1] - tvals[0]
    y_out = np.zeros((len(init_state), Nsteps), dtype=np.complex128)

    psi = init_state
    y_out[:, 0] = psi

    for i in range(1, Nsteps):
        t_mid = tvals[i - 1] + dt / 2
        lam = lam_func(t_mid, tau)
        dlam_dt = dlam_func(t_mid, tau)

        H = lam * (H1 + H3) + (1 - lam) * H2
        U = scipy.linalg.expm(-1j * dt * H)

        psi = U @ psi
        y_out[:, i] = psi

    return tvals, y_out
