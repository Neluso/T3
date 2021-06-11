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
    E_sam, E_ref_w, E_sam_w, freqs, n_s, k_s, theta_input = args
    H_teo = H_T3.H_sim_rouard_ref(freqs, n_s, k_s, params, theta_input)  # theta in air = 45 deg
    H_sam = E_sam_w / E_ref_w
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    delta_H_abs = np.abs(H_sam) - np.abs(H_teo)
    delta_H_phi = np.unwrap(np.angle(H_sam)) - np.unwrap(np.angle(H_teo))
    # plt.plot(np.unwrap(np.angle(H_sam)))
    # plt.plot(np.unwrap(np.angle(H_teo)))
    # plt.plot(delta_H_phi)
    # plt.show()
    # quit()
    delta_H_abs = delta_H_abs[10:60]
    delta_H_phi = delta_H_phi[10:40]
    return np.sum(delta_E**2)  # + np.sum(delta_H_abs**2) + 1e-2 * np.sum(delta_H_phi**2)


t_ref, E_ref = rd.read_1file('./data/aux_data/ref.txt')
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
plt.plot(np.unwrap(np.angle(E_ref_w)))
t_ref, E_ref = rd.read_1file('./data/20210609_PLA_PVA_2/ref1.txt')
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
plt.plot(np.unwrap(np.angle(E_ref_w)))
plt.show()
quit()

theta_input = 45  # deg
theta_input *= np.pi / 180

sample_num = 1
data_dir = './data/20210609_PLA_PVA_2/'

ref_file = data_dir + 'ref' + str(sample_num) + '.txt'
# ref_file = data_dir + 'elco_ref_244.txt'
# ref_file = './data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
E_ref *= -1
delta_t_ref = np.mean(np.diff(t_ref))
enlargement = 0 * E_ref.size
E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
t_centroid = t_ref[DSPf.centroid_E2(t_ref, E_ref)]
E_ref_sim = np.sin(2 * np.pi * t_ref * 0.35) * scipy.stats.norm.pdf(t_ref, loc=t_centroid, scale=1.1) / 0.3

# plt.plot(t_ref, E_ref)
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref2, E_ref_sim_w = DSPf.fourier_analysis(t_ref, E_ref_sim)
f_ref[0] = 1


n_celo = 1.5 * np.ones(f_ref.size)
alpha_celo = 0  # cm^-1 / THz
alpha_celo = alpha_celo * f_ref * 1e-12
k_celo = 1e2 * TDSC.c_0 * alpha_celo / (4 * np.pi * f_ref)

n_vid = 2.6 * np.ones(f_ref.size)
alpha_vid = 0  # cm^-1 / THz
alpha_vid = 20 * (f_ref * 1e-12)**2 + 60 * f_ref * 1e-12
k_vid = 1e2 * TDSC.c_0 * alpha_vid / (4 * np.pi * f_ref)


freq_aux, n_pes, n_pes_std, alpha_pes, alpha_pes_std = rd.read_from_1file('./data/20210609_PLA_PVA_1/calibration_polyester.txt')
n_pes = np.interp(f_ref, freq_aux, n_pes, left=n_pes[0], right=n_pes[-1])
alpha_pes = np.interp(f_ref, freq_aux, alpha_pes, left=alpha_pes[0], right=alpha_pes[-1])
k_pes = 1e2 * TDSC.c_0 * alpha_pes / (4 * np.pi * f_ref)


freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
# pol_n_PVA = np.polyfit(freq_aux, n_PVA, 1)
# n_PVA = pol_n_PVA[0] * f_ref + pol_n_PVA[1]
# pol_alpha_PVA = np.polyfit(freq_aux, alpha_PVA, 2)
# alpha_PVA = pol_alpha_PVA[0] * f_ref**2 + pol_alpha_PVA[1] * f_ref + pol_alpha_PVA[2]
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
# n_PVA = 1.89 * np.ones(f_ref.size)
# alpha_PVA = 52  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
k_PVA = 1e2 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)
k_PVA[0] = 0

freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
# pol_n_PLA = np.polyfit(freq_aux, n_PLA, 1)
# n_PLA = pol_n_PLA[0] * f_ref + n_PLA[1]
# pol_alpha_PLA = np.polyfit(freq_aux, alpha_PLA, 2)
# alpha_PLA = pol_alpha_PLA[0] * f_ref**2 + pol_alpha_PLA[1] * f_ref + pol_alpha_PLA[2]
n_PLA = np.interp(f_ref, freq_aux, n_PLA, left=n_PLA[0], right=n_PLA[-1])
alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=alpha_PLA[0], right=alpha_PLA[-1])
# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 28  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
k_PLA = 1e2 * TDSC.c_0 * alpha_PLA / (4 * np.pi * f_ref)
k_PLA[0] = 0


thick_tupl = (20e-6, 500e-6)
# thick_tupl = (0, 1)
correct_tuple = (0.95, 1.05)  # Error 10%

# k_bounds = [
#     (1e-6, 100e-6),
#     (100e-6, 200e-6)
#     , (-5e-4, 0)
#     # (20e-6, 350e-6)
#     # , (- 1000e-6, 0)
#     # , correct_tuple
#     # , correct_tuple
#     # , correct_tuple
#     # , correct_tuple
# ]

A_constraint = np.array(
    (
        (1, 1, 1)  # , 0, 0, 0, 0),
        # , (1, 0, -1)  #9 , 0, 0, 0, 0)
        # , (-1, 1, 0)
        # , (0, 1, -1)
    )
)
# A_constraint = spy_opt.LinearConstraint(A_constraint, (280e-6, -20e-6), (320e-6, 20e-6))
A_constraint = spy_opt.LinearConstraint(A_constraint, 280e-6, 320e-6)

k_bounds = [
    (0e-6, 3e-4),
    (0e-6, 3e-4),
    (0e-6, 3e-4)
]

k_bounds_test = [
    thick_tupl,
    (1, 3),
    (0, 1)

]

if __name__ == '__main__':
    sam_file = data_dir + 'sam' + str(sample_num) + '.txt'
    # sam_file = data_dir + 'elco_sam_244.txt'
    t_sam, E_sam = rd.read_1file(sam_file)
    E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
    t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
    plt.plot(t_sam, E_sam)
    n_s = np.array([TDSC.n_air * np.ones(n_PLA.shape), n_PLA, n_PVA, n_PLA])
    k_s = np.array([np.zeros(k_PLA.shape), k_PLA, k_PVA, k_PLA])
    # n_s = np.array([TDSC.n_air * np.ones(n_celo.shape), n_celo, n_vid])
    # k_s = np.array([np.zeros(k_celo.shape), k_celo, k_vid])
    # plt.plot(t_sam, np.fft.irfft(E_ref_w * H_T3.H_sim_rouard(f_ref, n_s, k_s, [0.7e-4, 1.6e-4, 0.7e-4])))
    t_sam *= 1e-12
    f_sam, E_sam_w = DSPf.fourier_analysis(t_sam, E_sam)
    plt.figure(30)
    plt.plot(f_ref * 1e-12, np.abs(E_sam_w / E_ref_w))  # * DSPf.wiener_filter(E_ref_w)))
    plt.figure(31)
    plt.plot(f_ref * 1e-12, np.abs(E_sam_w / E_ref_w))
    # plt.plot(f_ref * 1e-12, np.abs(H_T3.H_sim_rouard(f_ref, n_s, k_s, [0.7e-4, 1.6e-4, 0.7e-4])))
    plt.xlim([0, 1.5])
    plt.ylim([0, 1])
    # plt.show()
    # quit()

    num_statistics = 1
    z1 = list()
    z2 = list()
    z3 = list()
    for num_stats in range(num_statistics):
        # res = spy_opt.minimize(cost_function,
        #                        np.array((250e-6)),  # , 1, 1, 1, 1)),
        #                        # args=(E_sam, E_ref_w, E_sam_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA, theta_input),
        #                        args=(E_sam, E_ref_w, E_sam_w, f_ref, n_pes, k_pes, theta_input),
        #                        bounds=k_bounds,
        #                        # method='SLSQP',
        #                        # options={'ftol':1e-8}
        #                        # constraints=A_constraint
        #                        )
        res = spy_opt.differential_evolution(cost_function,
                                             k_bounds,
                                             args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s, theta_input),
                                             # args=(E_sam, E_ref_w, E_sam_w, f_ref, n_pes, k_pes, theta_input),
                                             # strategy='rand1exp',
                                             # tol=1e-8,
                                             # mutation=(0, 1.99),
                                             # recombination=0.4,
                                             # popsize=90,
                                             # maxiter=3000,
                                             updating='deferred',
                                             workers=-1,
                                             disp=True,  # step cost_function value
                                             polish=True
                                             # , constraints=A_constraint
                                             )
        # # print(res)
        # # print('-----------------------------------------------------')
        # res = spy_opt.minimize(cost_function,
        #                        res.x,  # , 1, 1, 1, 1)),
        #                        args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s, theta_input),
        #                        bounds=k_bounds,
        #                        method='SLSQP',
        #                        # options={'ftol':1e-8}
        #                        constraints=A_constraint
        #                        )
        #
        z1.append(res.x[0])
        z2.append(res.x[1])
        z3.append(res.x[2])
        print(res)

    # n_s = np.array([TDSC.n_air * np.ones(n_pes.shape), n_pes])
    # k_s = np.array([np.zeros(n_pes.shape), k_pes])
    # z1 = [80e-6]
    # z2 = [135e-6]
    # z3 = [80e-6]
    thick_s = np.array([np.mean(z1), np.mean(z2), np.mean(z3)])
    # thick_s = [125e-6, 190e-6, 125e-6]
    # n_s = [TDSC.n_air, np.mean(z2), TDSC.n_air]
    # k_s = [0, np.mean(z3) * f_ref * 1e-12, 0]
    # thick_s = np.array([30e-6, 180e-6, 0])
    H_teo = H_T3.H_sim_rouard_ref(f_ref, n_s, k_s, thick_s, theta_input)
    plt.figure(30)
    plt.plot(f_ref*1e-12, np.abs(H_teo))
    E_teo = np.fft.irfft(H_teo * E_ref_w)
    plt.figure(1)
    plt.plot(t_ref * 1e12, E_teo)
    # H_teo = H_T3.H_sim_rouard_ref_2_full(f_ref, n_s, k_s, thick_s, theta_input)
    # E_teo = np.fft.irfft(H_teo * E_ref_w)
    # plt.plot(t_ref * 1e12, E_teo)
    plt.xlim([0, 50])
    z1_mean = np.round(np.mean(z1) * 1e6, 0)
    z2_mean = np.round(np.mean(z2) * 1e6, 0)
    z3_mean = np.round(np.mean(z3) * 1e6, 0)
    print(z1_mean, z2_mean, z3_mean)
    print(z1_mean + z2_mean + z3_mean, '~~~ 300 um')
    plt.show()