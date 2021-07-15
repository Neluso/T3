import numpy as np
import scipy.interpolate as itpl
import scipy.stats

import H_T3
import TDS_constants as TDSC
import aux_functions
import read_data as rd
import matplotlib.pyplot as plt
import DSP_functions as DSPf
import scipy.optimize as spy_opt


def diffuse_layer():
    return 0


def cost_function(params, *args):
    E_sam, E_ref_w, E_sam_w, freqs, n_s, k_s = args
    H_teo = H_T3.H_sim_rouard(freqs, n_s, k_s, params)
    H_sam = E_sam_w / E_ref_w
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    return np.sum(delta_E**2)


ref_file = './data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
E_ref *= -1
delta_t_ref = np.mean(np.diff(t_ref))
enlargement = 10 * E_ref.size
E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref[0] = 1

n_1 = 1.4 * np.ones(f_ref.shape)
k_1 = 0.001 * f_ref * 1e-12
n_2 = 1.6 * np.ones(f_ref.shape)
k_2 = 0.001 * f_ref * 1e-12
k_adiab = 0.001 * f_ref * 1e-12
deg_in_air = np.pi * 45 / 180

n_s = np.array([TDSC.n_air * np.ones(n_1.shape), n_1, n_2])
k_s = np.array([np.zeros(k_1.shape), k_1, k_2])
thick_s = np.array([1000e-6, 1000e-6])
H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, deg_in_air)
plt.plot(np.fft.irfft(H_teo * E_ref_w))

n_s = np.array([TDSC.n_air * np.ones(n_1.shape), n_1, n_1 + 0.015, n_1 + 0.035, n_1 + 0.055, n_1 + 0.075, n_1 + 0.095,
                n_1 + 0.115, n_1 + 0.135, n_1 + 0.155, n_1 + 0.175, n_1 + 0.195, n_2])
k_s = np.array([np.zeros(k_1.shape), k_1, k_adiab, k_adiab, k_adiab, k_adiab, k_adiab,
                k_adiab, k_adiab, k_adiab, k_adiab, k_adiab, k_2])
thick_s = np.array([990e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 990e-6])

H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, deg_in_air)
plt.plot(np.fft.irfft(H_teo * E_ref_w))
plt.show()
