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
    H_sam = E_sam_w / E_ref_w
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    delta_E = E_sam - E_teo
    delta_H_abs = np.abs(H_sam) - np.abs(H_teo)
    delta_H_phi = np.unwrap(np.angle(H_sam)) - np.unwrap(np.angle(H_teo))
    # delta_H_abs = delta_H_abs[15:45]
    # delta_H_phi = delta_H_phi[15:45]
    return np.sum(delta_E**2)
    # return 0.01 * np.sum(delta_E**2) + np.sum(delta_H_abs**2)


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
f_ref[0] = 1

freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
# alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=0, right=alpha_PVA[-1])
# n_PVA = 1.89 * np.ones(f_ref.size)
# alpha_PVA = 52  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
k_PVA = 1e2 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)
print(k_PVA)
quit()

freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
n_PLA = np.interp(f_ref, freq_aux, n_PLA, left=n_PLA[0], right=n_PLA[-1])
# alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=alpha_PLA[0], right=alpha_PLA[-1])
alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=0, right=alpha_PLA[-1])
# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 28  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
k_PLA = 1e2 * TDSC.c_0 * alpha_PLA / (4 * np.pi * f_ref)


thick_tupl = (20e-6, 500e-6)
correct_tuple = (0.09, 1.01)  # Error 10%

k_bounds = [
    (1e-6, 135e-6),
    (1e-6, 205e-6),
    (1e-6, 135e-6)
]

A_constraint = np.array(
    (
        (1, 1, 1)
        , (1, 0, -1)
        # , (-1, 1, -1)
    )
)
A_constraint = spy_opt.LinearConstraint(A_constraint, (280e-6, -20e-6), (320e-6, 20e-6))
# A_constraint = spy_opt.LinearConstraint(A_constraint, 280e-6, 310e-6)
# A_constraint = spy_opt.LinearConstraint(A_constraint, (280e-6, -20e-6, 0), (320e-6, 20e-6, 300e-6))


if __name__ == '__main__':
    t_sam, E_sam = rd.read_1file('./data/PosV_75.000000_PosH_5.000000.txt')
    E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
    t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
    plt.plot(t_sam, E_sam, label='sam')
    n_s = np.array([TDSC.n_air * np.ones(n_PLA.shape), n_PLA, n_PVA, n_PLA, TDSC.n_air * np.ones(n_PLA.shape)])
    k_s = np.array([np.zeros(k_PLA.shape), k_PLA, k_PVA, k_PLA, np.zeros(k_PLA.shape)])
    t_sam *= 1e-12
    f_sam, E_sam_w = DSPf.fourier_analysis(t_sam, E_sam)

    num_statistics = 1
    z1 = list()
    z2 = list()
    z3 = list()
    for num_stats in range(num_statistics):
        res = spy_opt.differential_evolution(cost_function,
                                             k_bounds,
                                             args=(E_sam, E_ref_w, E_sam_w, f_ref, n_s, k_s),
                                             # strategy='rand1exp',
                                             # tol=1e-8,
                                             # mutation=(0, 1.99),
                                             # recombination=0.8,
                                             # popsize=90,
                                             # maxiter=3000,
                                             updating='deferred',
                                             workers=-1,
                                             disp=True,  # step cost_function value
                                             polish=True
                                             , constraints=A_constraint
                                             )
        z1.append(res.x[0])
        z2.append(res.x[1])
        z3.append(res.x[2])
        print(res)
    thick_s = np.array([np.mean(z1), np.mean(z2), np.mean(z3)])
    H_teo = H_T3.H_sim_rouard(f_ref, n_s, k_s, thick_s)
    E_teo = np.fft.irfft(H_teo * E_ref_w)
    plt.figure(1)
    plt.plot(t_ref * 1e12, E_teo, label='fit')
    plt.legend()
    z1_mean = np.round(np.mean(z1) * 1e6, 0)
    z2_mean = np.round(np.mean(z2) * 1e6, 0)
    z3_mean = np.round(np.mean(z3) * 1e6, 0)
    print(z1_mean, z2_mean, z3_mean)
    print(z1_mean + z2_mean + z3_mean, '~~~ 300 um')
    plt.show()
