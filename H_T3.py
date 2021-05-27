import numpy as np
import TDS_constants as TDSC


def ct(n_1, n_2):  # n_1: incident from, n_2: incident to
    return 2 * n_1 / (n_1 + n_2)


def cr(n_1, n_2):  # n_1: incident from, n_2: incident to
    return (n_1 - n_2) / (n_1 + n_2)


def phase_factor(n, k, thick, freq):  # theta in radians
    omg = 2 * np.pi * freq
    phi = omg * thick / TDSC.c_0
    return np.exp(- 1j * n * phi) * np.exp(- k * phi)


def fabry_perot(freq, n_i, k_i, thick_i, n_1, k_1, n_2, k_2):
    cri2 = cr(n_i, n_2)
    cri1 = cr(n_i, n_1)
    # cri2 = cr(n_i - 1j * k_i, n_2 - 1j * k_2)
    # cri1 = cr(n_i - 1j * k_i, n_1 - 1j * k_1)
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
