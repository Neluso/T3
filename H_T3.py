import numpy as np
import TDS_constants as TDSC
import matplotlib.pyplot as plt


def theta_from_air(theta_air, n_mat):
    return np.arcsin(TDSC.n_air * np.sin(theta_air) / n_mat)


def ct(n_1, n_2):  # n_1: incident from, n_2: incident to
    return 2 * n_1 / (n_1 + n_2)


def cr(n_1, n_2):  # n_1: incident from, n_2: incident to
    return (n_1 - n_2) / (n_1 + n_2)


def cts(n_1, n_2, theta_1, theta_2):  # n_1: incident from, n_2: incident to
    return 2 * n_1 * np.cos(theta_1) / (n_1 * np.cos(theta_1) + n_2 * np.cos(theta_2))


def crs(n_1, n_2, theta_1, theta_2):  # n_1: incident from, n_2: incident to
    return (n_1 * np.cos(theta_1) - n_2 * np.cos(theta_2)) / (n_1 * np.cos(theta_1) + n_2 * np.cos(theta_2))


def ctp(n_1, n_2, theta_1, theta_2):  # n_1: incident from, n_2: incident to
    return 2 * n_1 * np.cos(theta_1) / (n_1 * np.cos(theta_2) + n_2 * np.cos(theta_1))


def crp(n_1, n_2, theta_1, theta_2):  # n_1: incident from, n_2: incident to
    return (n_2 * np.cos(theta_1) - n_1 * np.cos(theta_2)) / (n_1 * np.cos(theta_2) + n_2 * np.cos(theta_1))


def phase_factor(n, k, thick, freq):  # theta in radians
    omg = 2 * np.pi * freq
    phi = omg * thick / TDSC.c_0
    exp = np.exp(- 1j * n * phi)
    exp *= np.exp(- k * phi)
    return exp


# def


def fabry_perot(freq, n_i, k_i, thick_i, n_1, k_1, n_2, k_2):
    # cri2 = cr(n_i, n_2)
    # cri1 = cr(n_i, n_1)
    cri2 = cr(n_i - 1j * k_i, n_2 - 1j * k_2)
    cri1 = cr(n_i - 1j * k_i, n_1 - 1j * k_1)
    exp_phi = phase_factor(n_i, k_i, 2 * thick_i, freq)
    fp = 1 - cri2 * cri1 * exp_phi
    fp = 1 / fp
    # fp = 1 + cri2 * cri1 * exp_phi
    return fp


def H_sim(freq, n_i, k_i, thick_i, n_1, k_1, n_2, k_2):
    exp_phi = phase_factor(n_i - TDSC.n_air, k_i, thick_i, freq)
    ct1i = ct(n_1, n_i)
    cti2 = ct(n_i, n_2)
    # ct1i = ct(n_1 - 1j * k_1, n_i - 1j * k_i)
    # cti2 = ct(n_i - 1j * k_i, n_2 - 1j * k_2)
    fp = fabry_perot(freq, n_i, k_i, thick_i, n_1, k_1, n_2, k_2)
    H_i = ct1i * cti2 * exp_phi * fp
    return H_i


def H_sim_rouard(freq, n_s, k_s, thick_s):  # n_s y k_s sandwitch de n_air
    cn_s = n_s - 1j * k_s
    H_teo = ct(cn_s[0], cn_s[1])

    for layer in range(1, len(thick_s) + 1):
        phil = phase_factor(n_s[layer] - TDSC.n_air, k_s[layer], thick_s[layer - 1], freq)
        ctll1 = ct(cn_s[layer], cn_s[layer + 1])
        fpl = fabry_perot(freq, n_s[layer], k_s[layer], thick_s[layer - 1],
                          n_s[layer - 1], k_s[layer - 1], n_s[layer + 1], k_s[layer + 1])
        H_teo = H_teo * phil * ctll1 * fpl
    return H_teo


def H_sim_rouard_ref(freq, n_s, k_s, thick_s, theta_in_air, pol='s'):  # n_s y k_s start n_air
    H_teo = - 1  # refs medidas en reflexion, por lo tanto hay que corregir phase de pi
    layers = len(thick_s)
    cn_s = n_s - 1j * k_s
    for layer in range(1, layers + 1):
        layer = 1 + layers - layer
        theta_l = theta_from_air(theta_in_air, n_s[layer])
        theta_l1 = theta_from_air(theta_in_air, n_s[layer - 1])
        if pol == 's':
            crl1l = crs(cn_s[layer - 1], cn_s[layer], theta_l1, theta_l)
            ctl1l = cts(cn_s[layer - 1], cn_s[layer], theta_l1, theta_l)
            ctll1 = cts(cn_s[layer], cn_s[layer - 1], theta_l, theta_l1)
        elif pol == 'p':
            crl1l = crp(cn_s[layer - 1], cn_s[layer], theta_l1, theta_l)
            ctl1l = ctp(cn_s[layer - 1], cn_s[layer], theta_l1, theta_l)
            ctll1 = ctp(cn_s[layer], cn_s[layer - 1], theta_l, theta_l1)
        thick_l = 2 * thick_s[layer - 1] * np.cos(theta_l)
        phil = phase_factor(n_s[layer], k_s[layer], thick_l, freq)
        trans_term = ctl1l * ctll1 * H_teo * phil  # transmission term
        fp_term = 1 + crl1l * H_teo * phil  # fabry-pérot term
        H_teo = crl1l + trans_term / fp_term
    return H_teo * phase_factor(- TDSC.n_air, 0, 2 * np.sum(thick_s) * np.cos(theta_in_air), freq)


def H_sim_rouard_ref_2_full(freq, n_s, k_s, thick_s, theta_in_air, pol='s'):  # n_s y k_s start n_air
    H_teo = 1  # refs medidas en reflexion, por lo tanto hay que corregir phase de pi
    n_i = n_s[2]
    k_i = k_s[2]
    n_o = n_s[1]
    k_o = k_s[1]
    cn_i = n_i - 1j * k_i
    cn_o = n_o - 1j * k_o
    thick_i = thick_s[1]
    thick_o = thick_s[0]
    d_air = thick_s[2]
    theta_i = theta_from_air(theta_in_air, n_i)
    theta_o = theta_from_air(theta_in_air, n_o)

    cr_oi = crs(cn_o, cn_i, theta_o, theta_i)
    ct_oi = cts(cn_o, cn_i, theta_o, theta_i)
    ct_io = cts(cn_i, cn_o, theta_i, theta_o)
    phi_i = phase_factor(n_i, k_i, 2 * thick_i * np.cos(theta_i), freq)


    cr_ao = crs(TDSC.n_air, cn_o, theta_in_air, theta_o)
    ct_ao = cts(TDSC.n_air, cn_o, theta_in_air, theta_o)
    ct_oa = cts(cn_o, TDSC.n_air, theta_o, theta_in_air)
    phi_o = phase_factor(n_o, k_o, 2 * thick_o * np.cos(theta_o), freq)

    H_teo = crs(cn_i, 1e20, theta_i, 0) + cr_ao - cr_oi

    trans_term_i = ct_oi * ct_io * H_teo * phi_i
    fp_term_i = 1 + cr_oi * H_teo * phi_i
    H_teo = cr_oi + trans_term_i / fp_term_i
    trans_term_o = ct_ao * ct_oa * H_teo * phi_o
    fp_term_o = 1 + cr_ao * H_teo * phi_o
    H_teo = cr_ao + trans_term_o / fp_term_o

    return - H_teo * phase_factor(TDSC.n_air, 0, 2 * d_air * np.cos(theta_in_air), freq)


def plot_coeffs(n_1, n_2):
    rads = np.arange(5001) * np.pi / 10000
    np.arcsin(n_1 * np.sin(rads) / n_2)
    degs = rads * 180 / np.pi
    theta_B = np.arctan(n_2 / n_1)
    theta_c = np.arcsin(n_2 / n_1)
    deg_B = theta_B * 180 / np.pi
    deg_c = theta_c * 180 / np.pi
    plt.plot(degs, crs(n_1, n_2, rads, np.arcsin(n_1 * np.sin(rads) / n_2)), 'r-', label=r'$r_s$')
    plt.plot(degs, cts(n_1, n_2, rads, np.arcsin(n_1 * np.sin(rads) / n_2)), 'r--', label=r'$t_s$')
    plt.plot(degs, crp(n_1, n_2, rads, np.arcsin(n_1 * np.sin(rads) / n_2)), 'b-', label=r'$r_p$')
    plt.plot(degs, ctp(n_1, n_2, rads, np.arcsin(n_1 * np.sin(rads) / n_2)), 'b--', label=r'$t_p$')
    plt.vlines(deg_B, -1, 3, linestyles='dotted', label=r'$\theta_B$ = '+str(round(deg_B, 1)))
    plt.vlines(deg_c, -1, 3, linestyles='dashed', label=r'$\theta_c$ = '+str(round(deg_c, 1)))
    plt.legend()
    plt.xlim([0, 90])
    plt.ylim([-1, 3])
    plt.show()
    return 0
