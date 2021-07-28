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


def add_diffuse_layer(array_in, add_array, idx_to_add):
    array_out = list()
    for i in range(array_in.shape[0]):
        if i == idx_to_add:
            for j in range(add_array.shape[0]):
                array_out.append(add_array[j])
        array_out.append(array_in[i])
    return np.array(array_out)


def cost_function(params, *args):
    E_sam, E_ref_w, E_sam_w, freqs, n_s, k_s, deg_in = args
    H_teo = H_T3.H_sim_rouard_ref(freqs, n_s, k_s, params, deg_in)
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    return np.sum(delta_E**2)


def cost_function_adiabatica(params, *args):
    E_sam, E_ref_w, E_sam_w, freqs, n_s, k_s, deg_in, N_grid = args
    thick_s = np.array([])
    if params.size <= 2:
        thick_s = params
    else:
        for i in range(params.size):
            if i%2 == 0:
                thick_s = np.append(thick_s, params[i])
            else:
                difuse_thick = np.ones(N_grid) * params[i] / N_grid
                for j in range(difuse_thick.size):
                    thick_s = np.append(thick_s, difuse_thick[j])
    H_teo = H_T3.H_sim_rouard_ref(freqs, n_s, k_s, thick_s, deg_in)
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    return np.sum(delta_E**2)


ref_file = './data/ref.txt'
# sam_file = './data/adiab_2_lay_200_tran_10.txt'
sam_file = './data/noad_2_lay_200.txt'
t_ref, E_ref = rd.read_1file(ref_file)
E_ref *= -1
delta_t_ref = np.mean(np.diff(t_ref))
enlargement = 0 * E_ref.size
E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref[0] = 1
t_sam, E_sam = rd.read_1file(sam_file)
E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
t_sam *= 1e-12
f_sam, E_sam_w = DSPf.fourier_analysis(t_sam, E_sam)
f_sam[0] = 1


n_1 = 1.45 * np.ones(f_ref.shape)
k_1 = np.zeros(f_ref.size)  # np.zeros(f_ref.size)  # 0.001 * f_ref * 1e-12
n_2 = 1.55 * np.ones(f_ref.shape)
k_2 = np.zeros(f_ref.size)
n_3 = 1.8 * np.ones(f_ref.shape)
k_3 = 0.001 * f_ref * 1e-12
k_adiab = 0.001 * f_ref * 1e-12
deg_in_air = np.pi * 45 / 180
N_grid = 10
m1 = (n_2 - n_1) / N_grid
m2 = (n_3 - n_2) / N_grid

difuse_n1 = list()
difuse_n2 = list()
difuse_k = list()
for i in range(N_grid):
    difuse_n1.append(n_1 + m1 * (i + 0.5))
    difuse_n2.append(n_2 + m2 * (i + 0.5))
    difuse_k.append(k_adiab)

difuse_n1 = np.array(difuse_n1)
difuse_n2 = np.array(difuse_n2)
difuse_k = np.array(difuse_k)


n_s = np.array([TDSC.n_air * np.ones(n_1.shape), n_1, n_2])  # , n_3])
k_s = np.array([np.zeros(k_1.shape), k_1, k_2])  # , k_3])
H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, np.array([100e-6, 100e-6]), deg_in_air)
E_noad = np.fft.irfft(H_teo * E_ref_w)
# plt.plot(np.fft.irfft(H_teo * E_ref_w), label='noad', lw=1)
# thick_s = np.array([200e-6, 10e-6, 200e-6])
# n_s = add_diffuse_layer(n_s, difuse_n2, 3)
# n_s = add_diffuse_layer(n_s, difuse_n1, 2)
# k_s = add_diffuse_layer(k_s, difuse_k, 3)
# k_s = add_diffuse_layer(k_s, difuse_k, 2)


# cost_function_adiabatica(np.array([95e-6, 10e-6, 95e-6]), E_noad, E_ref_w, E_sam_w, f_ref, n_s, k_s, deg_in_air, N_grid)
# plt.show()
# quit()


k_bounds = [
    (1e-6, 400e-6),
    (1e-6, 400e-6)
]
res = spy_opt.differential_evolution(cost_function,
                                     k_bounds,
                                     args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s, deg_in_air)
                                     )
print('Fit normal')
print(res.x * 1e6, 'um')

# n_s = add_diffuse_layer(n_s, difuse_n2, 3)
n_s = add_diffuse_layer(n_s, difuse_n1, 2)
# k_s = add_diffuse_layer(k_s, difuse_k, 3)
k_s = add_diffuse_layer(k_s, difuse_k, 2)
k_bounds = [
    (150e-6, 220e-6),
    (1e-6, 20e-6),
    (190e-6, 220e-6)
]
res = spy_opt.differential_evolution(cost_function_adiabatica,
                                     k_bounds,
                                     args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s, deg_in_air, N_grid)
                                     )
print('Fit adiabático')
print(res.x * 1e6, 'um')


# difuse_thick = np.ones(N_grid) * 1e-6 / N_grid
# thick_s = add_diffuse_layer(np.array([199.6e-6, 199.6e-6, 199.6e-6]), difuse_thick, 2)
# thick_s = add_diffuse_layer(thick_s, difuse_thick, 1)
# H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, deg_in_air)
# plt.plot(np.fft.irfft(H_teo), label='ad_1', lw=1)
#
# difuse_thick = np.ones(N_grid) * 1e-5 / N_grid
# thick_s = add_diffuse_layer(np.array([196.66e-6, 196.66e-6, 196.66e-6]), difuse_thick, 2)
# thick_s = add_diffuse_layer(thick_s, difuse_thick, 1)
# H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, deg_in_air)
# plt.plot(np.fft.irfft(H_teo), label='ad_10', lw=1)
#
# # difuse_thick = np.ones(N_grid) * 2e-5 / N_grid
# # thick_s = add_diffuse_layer(np.array([980e-6, 980e-6, 980e-6]), difuse_thick, 2)
# # thick_s = add_diffuse_layer(thick_s, difuse_thick, 1)
# # H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, deg_in_air)
# # plt.plot(np.fft.irfft(H_teo), label='ad_20', lw=1)
#
# plt.legend()
# plt.show()
