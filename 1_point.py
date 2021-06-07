import numpy as np
import scipy.interpolate as itpl
import H_T3
import TDS_constants as TDSC
import read_data as rd
import os
import matplotlib.pyplot as plt
import DSP_functions as DSPf
import scipy.optimize as spy_opt
from mpl_toolkits.mplot3d import Axes3D
import time


def param_2_thickess(param_d):
    return (35 * param_d) * 1e-5  # 0 to 350 um


def abs_diff(params):
    x1, x2, x3, aux1, aux2, aux3, aux4 = params
    return np.abs(x1-x3)


def cost_function(params, *args):
    E_sam, E_ref_w, freqs, n_PVA, k_PVA, n_PLA, k_PLA = args
    # thick_1, thick_2, thick_3, d_air = params
    # thick_1, thick_2, thick_3 = params
    thick_1, thick_2, thick_3, delta_1, delta_2, delta_3, delta_4 = params
    # thick_2 = param_2_thickess(thick_2)
    # thick_1 = param_2_thickess(thick_1)
    # thick_3 = param_2_thickess(thick_3)
    n_PVA *= delta_1
    k_PVA *= delta_2
    n_PLA *= delta_3
    k_PLA *= delta_4

    n_s = [TDSC.n_air, n_PLA, n_PVA, n_PLA, TDSC.n_air]
    k_s = [0, k_PLA, k_PVA, k_PLA, 0]
    thick_s = [thick_1, thick_2, thick_3]

    H_teo = H_T3.H_sim_rouard(freqs, n_s, k_s, thick_s)
    w_filt = DSPf.wiener_filter(E_ref_w)
    H_teo *= w_filt
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    # return np.sum((E_sam - E_teo) ** 2)
    return np.sum(np.abs(E_sam - E_teo))


samples_num = 6000
ref_file = './data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
# t_ref = t_ref[:samples_num]
# E_ref = E_ref[:samples_num]
delta_t_ref = np.mean(np.diff(t_ref))

enlargement = 0 * E_ref.size
E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
# t_ref = - np.flip(t_ref)
# print(t_ref)
# quit()
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref[0] = 1

# n_PVA = 1.89 * np.ones(f_ref.size)
# alpha_PVA = 52  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
# plt.figure(1)
# plt.plot(f_ref, n_PVA)
# plt.figure(2)
# plt.plot(f_ref, alpha_PVA)
freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
# plt.plot(freq_aux, n_PVA)
# max_idx = np.where(f_ref <= freq_aux[-1])[0][-1]
# n_PVA = np.interp(f_ref[:max_idx], freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
# plt.plot(f_ref[:max_idx], n_PVA)
# plt.show()
# quit()
# alpha_PVA = np.interp(f_ref[:max_idx], freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
pol_n_PVA = np.polyfit(freq_aux, n_PVA, 1)
n_PVA = pol_n_PVA[0] * f_ref + pol_n_PVA[1]
pol_alpha_PVA = np.polyfit(freq_aux, alpha_PVA, 1)
alpha_PVA = pol_alpha_PVA[0] * f_ref + pol_alpha_PVA[1]
k_PVA = 1e-10 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)

# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 28  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
max_idx = np.where(f_ref <= freq_aux[-1])[0][-1]
n_PLA = np.interp(f_ref, freq_aux, n_PLA, left=n_PLA[0], right=n_PLA[-1])
alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=alpha_PLA[0], right=alpha_PLA[-1])
k_PLA = 1e-10 * TDSC.c_0 * alpha_PLA / (4 * np.pi * f_ref)

# plt.figure(1)
# plt.plot(f_ref, n_PLA, f_ref, n_PVA)
# plt.figure(2)
# plt.plot(f_ref, alpha_PLA, f_ref, alpha_PVA)
# plt.show()


# plt.plot(f_ref, n_PLA)
# plt.show()


# thick_tupl = (20e-6, 350e-6)
thick_tupl = (20e-6, 1e-3)
correct_tuple = (0.9, 1.1)  # Error 10%

k_bounds = [
    thick_tupl,
    thick_tupl,
    thick_tupl
    # , (- 1000e-6, 0)
    , correct_tuple
    , correct_tuple
    , correct_tuple
    , correct_tuple
]


if __name__ == '__main__':
    t_sam, E_sam = rd.read_1file('./data/PosV_70.000000_PosH_10.000000.txt')
    # t_sam = t_sam[:samples_num]
    # E_sam = E_sam[:samples_num]
    E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
    t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
    plt.plot(t_sam, E_sam)
    t_sam *= 1e-12

    num_statistics = 3
    z1 = list()
    z2 = list()
    z3 = list()
    for num_stats in range(num_statistics):
        # res = spy_opt.minimize(cost_function,
        #                        np.array((70e-6, 100e-6, 70e-6, 1, 1, 1, 1)),
        #                        args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
        #                        bounds=k_bounds,
        #                        method='L-BFGS-B',
        #                        options={'ftol':1e-8}
        #                        # , constraints=[spy_opt.NonlinearConstraint(abs_diff, 0, 20e-6)]
        #                        )
        res = spy_opt.differential_evolution(cost_function,
                                             k_bounds,
                                             args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
                                             # strategy='rand1exp',
                                             # tol=1e-8,
                                             # mutation=(0.5, 1.5),
                                             # recombination=0.1,
                                             # bounds=k_bounds,
                                             popsize=30,
                                             # maxiter=3000,
                                             updating='deferred',
                                             workers=-1,
                                             disp=True,  # step cost_function value
                                             polish=False
                                             # , constraints=[spy_opt.NonlinearConstraint(abs_diff, 0, 20e-6)]
                                             )
        z1.append(res.x[0])
        z2.append(res.x[1])
        z3.append(res.x[2])
        print(res)
        # z1.append(param_2_thickess(res.x[0]))
        # z2.append(param_2_thickess(res.x[2]))
        # z3.append(param_2_thickess(res.x[2]))
    # H1 = H_T3.H_sim(f_ref, n_PLA, k_PLA, np.mean(z1), TDSC.n_air, 0, n_PVA, k_PVA)
    # H2 = H_T3.H_sim(f_ref, n_PVA, k_PVA, np.mean(z2), n_PLA, k_PLA, n_PLA, k_PLA)
    # H3 = H_T3.H_sim(f_ref, n_PLA, k_PLA, np.mean(z3), n_PVA, k_PVA, TDSC.n_air, 0)
    # air_phase = H_T3.phase_factor(TDSC.n_air, 0, - (np.mean(z1) + np.mean(z2) + np.mean(z3)), f_ref)
    # E_teo = np.fft.irfft(H1 * H2 * H3 * E_ref_w)
    print(np.mean(z1))
    n_s = [TDSC.n_air, n_PLA, n_PVA, n_PLA, TDSC.n_air]
    k_s = [0, k_PLA, k_PVA, k_PLA, 0]
    thick_s = [np.mean(z1), np.mean(z2), np.mean(z3)]
    H_teo = H_T3.H_sim_rouard(f_ref, n_s, k_s, thick_s)
    E_teo = np.fft.irfft(H_teo * E_ref_w)
    plt.figure(1)
    plt.plot(t_ref * 1e12, E_teo)
    z1_mean = np.round(np.mean(z1) * 1e6, 0)
    z2_mean = np.round(np.mean(z2) * 1e6, 0)
    z3_mean = np.round(np.mean(z3) * 1e6, 0)
    print(z1_mean, z2_mean, z3_mean)
    print(z1_mean + z2_mean + z3_mean, '--- 610 um')
    plt.show()
    # point = (posV, posH, z1[0], z2[0], z3[0])
    # print(point
    # print((posV, posH, res.x[1], res.x[2], res.x[3]))
    # x.append(posH)
    # y.append(posV)
    # zs1.append(res.x[1])
    # zs2.append(res.x[2])
    # zs3.append(res.x[3])
