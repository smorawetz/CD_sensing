from time_evol_open_sys import simulate_LMG_CD


def evol_rho(S, chi, g, h, tau, order, Omega, Gamma, gamma_el, n_save):
    rho = simulate_LMG_CD(
        N=int(2 * S), chi=chi, h=h, g=g, tau=tau, agp_order=order,
        Omega=Omega, Gamma=Gamma, gamma_el=gamma_el, n_save=n_save,
    )
    return rho.full()
