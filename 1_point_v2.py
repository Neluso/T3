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


def abs_diff(params):
    x1, x2, x3, aux1, aux2, aux3, aux4 = params
    # return np.abs(param_2_thickess(x1) - param_2_thickess(x3))
    return x1 + x2 + x3


def param_2_thickess(param_d):
    return 20 * (20 * param_d + 1) * 1e-6  # 0 to 350 um


def cost_function(params, *args):
    E_sam, E_ref_w, E_sam_w, freqs, n_s, k_s = args
    H_teo = H_T3.H_sim_rouard(freqs, n_s, k_s, params)
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    return np.sum(delta_E**2)


ref_file = './data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
delta_t_ref = np.mean(np.diff(t_ref))
enlargement = 0 * E_ref.size
E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
t_centroid = t_ref[DSPf.centroid_E2(t_ref, E_ref)]
E_ref_sim = np.sin(2 * np.pi * t_ref * 0.35) * scipy.stats.norm.pdf(t_ref, loc=t_centroid, scale=1.1) / 0.3

t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref2, E_ref_sim_w = DSPf.fourier_analysis(t_ref, E_ref_sim)
f_ref[0] = 1

freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
# n_PVA = 1.89 * np.ones(f_ref.size)
# alpha_PVA = 52  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
k_PVA = 1e2 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)

freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 28  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
k_PLA = 1e2 * TDSC.c_0 * alpha_PLA / (4 * np.pi * f_ref)


thick_tupl = (20e-6, 500e-6)
# thick_tupl = (0, 1)
correct_tuple = (0.09, 1.01)  # Error 10%

k_bounds = [
    (1e-6, 135e-6),
    (1e-6, 205e-6),
    (1e-6, 135e-6)
    # , (- 1000e-6, 0)
    # , (correct_tuple)
    # , correct_tuple
    # , correct_tuple
    # , correct_tuple
]

A_constraint = np.array(
    (
        (1, 1, 1),  # , 0, 0, 0, 0),
        (1, 0, -1)  # , 0, 0, 0, 0)
        # , (-1, 1, 0)
        # , (0, 1, -1)
    )
)
A_constraint = spy_opt.LinearConstraint(A_constraint, (280e-6, -20e-6), (320e-6, 20e-6))

# k_bounds = [
#     (1e-6, 1e-3),
#     (1e-6, 1e-3),
#     (1e-6, 1e-3)
#     , (1.8, 2.2)
#     , (0.01, 0.9)
#     , (1.4, 1.8)
#     , (0.01, 0.9)
# ]

k_bounds_test = [
    thick_tupl,
    (1, 3),
    (0, 1)

]

if __name__ == '__main__':
    t_sam, E_sam = rd.read_1file('./data/PosV_75.000000_PosH_5.000000.txt')
    E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
    t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
    plt.plot(t_sam, E_sam)
    n_s = [TDSC.n_air, n_PLA, n_PVA, n_PLA, TDSC.n_air]
    k_s = [0, k_PLA, k_PVA, k_PLA, 0]
    # plt.plot(t_sam, np.fft.irfft(E_ref_w * H_T3.H_sim_rouard(f_ref, n_s, k_s, [0.7e-4, 1.6e-4, 0.7e-4])))
    t_sam *= 1e-12
    f_sam, E_sam_w = DSPf.fourier_analysis(t_sam, E_sam)
    plt.figure(30)
    plt.plot(f_ref * 1e-12, np.abs(E_sam_w / E_ref_w))  # * DSPf.wiener_filter(E_ref_w)))
    # plt.plot(f_ref * 1e-12, np.abs(H_T3.H_sim_rouard(f_ref, n_s, k_s, [0.7e-4, 1.6e-4, 0.7e-4])))
    # plt.xlim([0, 1.5])
    # plt.ylim([0, 1])
    # plt.show()
    # quit()

    num_statistics = 1
    z1 = list()
    z2 = list()
    z3 = list()
    for num_stats in range(num_statistics):
        # res = spy_opt.minimize(cost_function,
        #                        np.array((100e-6, 100e-6, 100e-6)),  # , 1, 1, 1, 1)),
        #                        args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
        #                        bounds=k_bounds,
        #                        method='SLSQP',
        #                        # options={'ftol':1e-8}
        #                        constraints=[spy_opt.LinearConstraint(A_constraint, (265e-6, -20e-6), (335e-6, 20e-6))]
        #                        )
        res = spy_opt.differential_evolution(cost_function,
                                             k_bounds,
                                             args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s),
                                             # strategy='rand1exp',
                                             # tol=1e-8,
                                             # mutation=(0, 1.99),
                                             # recombination=0.4,
                                             # popsize=30,
                                             # maxiter=3000,
                                             updating='deferred',
                                             workers=-1,
                                             disp=True,  # step cost_function value
                                             polish=True
                                             # , constraints=A_constraint
                                             )
        # print(res)
        # print('-----------------------------------------------------')
        # res = spy_opt.minimize(cost_function,
        #                        res.x,  # , 1, 1, 1, 1)),
        #                        args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
        #                        bounds=k_bounds,
        #                        method='SLSQP',
        #                        # options={'ftol':1e-8}
        #                        constraints=spy_opt.LinearConstraint(A_constraint, (296e-6, 0e-6), (317e-6, 0e-6))
        #                        )

        z1.append(res.x[0])
        z2.append(res.x[1])
        z3.append(res.x[2])
        print(res)
    n_s = [TDSC.n_air, n_PLA, n_PVA, n_PLA, TDSC.n_air]
    k_s = [0, k_PLA, k_PVA, k_PLA, 0]
    thick_s = [np.mean(z1), np.mean(z2), np.mean(z3)]
    # thick_s = [100e-6, 190e-6, 100e-6]
    # n_s = [TDSC.n_air, np.mean(z2), TDSC.n_air]
    # k_s = [0, np.mean(z3) * f_ref * 1e-12, 0]
    # thick_s = [np.mean(z1)]
    H_teo = H_T3.H_sim_rouard(f_ref, n_s, k_s, thick_s)
    plt.figure(30)
    plt.plot(f_ref[:60]*1e-12, np.abs(H_teo[:60]))
    E_teo = np.fft.irfft(H_teo * E_ref_w)
    plt.figure(1)
    plt.plot(t_ref * 1e12, E_teo)
    z1_mean = np.round(np.mean(z1) * 1e6, 0)
    z2_mean = np.round(np.mean(z2) * 1e6, 0)
    z3_mean = np.round(np.mean(z3) * 1e6, 0)
    print(z1_mean, z2_mean, z3_mean)
    print(z1_mean + z2_mean + z3_mean, '~~~ 300 um')
    plt.show()