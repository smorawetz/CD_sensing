from pathlib import Path

import numpy as np
import qutip as qt
from qutip import piqs

optimal_gap_ratios = np.loadtxt(Path(__file__).with_name("universal_data.txt"))[:, 1]


def comp_AGP_coeffs(Omega, order, n_fit=10000):
    if order == 0:
        return np.empty(0)

    x = np.linspace(optimal_gap_ratios[order - 1], 1.0, n_fit)
    design = np.polynomial.chebyshev.chebvander(x, 2 * order - 1)[:, 1::2]
    coeffs = np.linalg.lstsq(design, -1.0 / x, rcond=None)[0]
    return coeffs / Omega


def lambda_ramp(t, tau):
    x = np.pi * t / (2.0 * tau)
    return np.sin(0.5 * np.pi * np.sin(x) ** 2) ** 2


def lambda_dot(t, tau):
    x = np.pi * t / (2.0 * tau)
    return (
        np.pi ** 2 / (4.0 * tau)
        * np.sin(np.pi * np.sin(x) ** 2)
        * np.sin(np.pi * t / tau)
    )


def commutator(A, B):
    return A * B - B * A


def build_chebyshev_agp(H, dH_dlambda, coeffs, Omega):
    """Build i sum_k c_k T_{2k+1}(ad_H/Omega)(dH/dlambda)."""
    if len(coeffs) == 0:
        return 0.0 * H

    H_scaled = H / Omega
    T_prev = dH_dlambda
    T_curr = commutator(H_scaled, dH_dlambda)
    polynomial = coeffs[0] * T_curr

    for degree in range(2, 2 * len(coeffs)):
        T_prev, T_curr = T_curr, 2.0 * commutator(H_scaled, T_curr) - T_prev
        if degree % 2 == 1:
            polynomial += coeffs[degree // 2] * T_curr

    A = 1j * polynomial
    return 0.5 * (A + A.dag())


def simulate_LMG_CD(
    N, chi, h, g, tau, agp_order, Omega, Gamma, gamma_el, n_save=2, return_result=False
):
    cheb_coeffs = comp_AGP_coeffs(Omega, agp_order)

    Sx = piqs.jspin(N, "x")
    Sz = piqs.jspin(N, "z")
    S = N / 2.0

    H0 = -(chi / (2 * np.sqrt(S * (S + 1)))) * (Sx * Sx) - h * Sx
    H1 = -g * Sz
    dH_dlambda = H0 - H1

    _, psi0 = H1.groundstate()
    rho0 = psi0.proj()

    L_diss = piqs.Dicke(
        N=N, collective_emission=Gamma / S, dephasing=gamma_el
    ).lindbladian()

    def H_of_t(t):
        lam = lambda_ramp(t, tau)
        H = lam * H0 + (1.0 - lam) * H1

        if agp_order == 0:
            return H

        A = build_chebyshev_agp(H, dH_dlambda, cheb_coeffs, Omega)
        return H + lambda_dot(t, tau) * A

    result = qt.mesolve(
        qt.QobjEvo(H_of_t), rho0, np.linspace(0.0, tau, n_save), c_ops=[L_diss], e_ops=None
    )

    if return_result:
        return result.final_state, result
    return result.final_state
