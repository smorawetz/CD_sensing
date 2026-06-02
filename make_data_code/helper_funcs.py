import numpy as np
import matplotlib.pyplot as plt
import scipy
import quspin

#############################################################
### auxiliary functions used to simulating time evolution ###
#############################################################


def cons_ops(S):
    if S % 1 == 0:
        Sstr = str(int(S))
    else:
        Sstr = str(int(2 * S)) + "/2"

    basis = quspin.basis.spin_basis_1d(L=1, S=Sstr)
    Sp = quspin.operators.hamiltonian(
        [["+", [[1.0, 0]]]],
        [],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
    ).toarray()
    Sm = quspin.operators.hamiltonian(
        [["-", [[1.0, 0]]]],
        [],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
    ).toarray()
    Sz = quspin.operators.hamiltonian(
        [["z", [[1.0, 0]]]],
        [],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
    ).toarray()
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2j)
    return Sx, Sy, Sz


def make_H(t, S, H_params, Smats, lam_func, lam_func_args):
    Sx, Sy, Sz = Smats
    chi, g, h = H_params
    lam = lam_func(t, *lam_func_args)
    return lam * (-chi / 2 / np.sqrt(S * (S + 1)) * Sz @ Sz - h * Sz) + (1 - lam) * (
        -g * Sx
    )


# ramps
def lin_lam_func(t, tau):
    return t / tau


def dlin_lam_func(t, tau):
    return 1.0 / tau


def smooth_lam_func(t, tau):
    return np.sin(np.pi / 2 * np.sin(np.pi / 2 * t / tau) ** 2) ** 2


def dsmooth_lam_func(t, tau):
    v = np.sin(np.pi / 2 * t / tau) ** 2
    dv_dt = (np.pi / (2 * tau)) * np.sin(np.pi * t / tau)
    return (np.pi / 2) * np.sin(np.pi * v) * dv_dt


# getting initial ground state
def get_GS(t, S, H_params, lam_func, lam_func_args):
    Smats = cons_ops(S)
    # init_params = (0, H_params[1], 0)  # assume init state ONLY sees transverse field
    # init_params = (H_params[0], H_params[1], 0)  # assume init state DOES NOT see long. field
    init_params = H_params  # assume init state is ground state of initial H
    H = make_H(t, S, init_params, Smats, lam_func, lam_func_args)
    evals, evecs = np.linalg.eigh(H)
    gstate = evecs[:, 0]
    return gstate


# making polynomial fit for universal AGP
def fit_func(x, *coeffs):
    full_coeffs = np.zeros(2 * len(coeffs))
    full_coeffs[1::2] = coeffs
    return np.polynomial.chebyshev.chebval(x, full_coeffs)


def comp_AGP_coeffs(Delta, Omega, order):
    x = np.linspace(Delta, Omega, 10000)
    y = -1 / x
    opt_coeffs, _ = scipy.optimize.curve_fit(fit_func, x, y, p0=np.zeros(order))
    return opt_coeffs


def make_fname(S, chi, g, h, tau, order, Omega):
    return f"S{S}_chi{chi}_g{g}_h{h}_tau{tau}_order{order}_Omega{Omega}.txt"
