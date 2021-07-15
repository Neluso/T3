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
import tqdm


def param_2_thickess(param_d):
    return (25 * param_d + 5) * 1e-5  # 50 to 250 um


def cost_function(params, *args):
    E_sam, E_ref_w, freqs, n_s, k_s = args
    if (params < np.zeros(params.shape)).any():
        return np.sum(np.abs(E_sam))
    H_teo = H_T3.H_sim_rouard(freqs, n_s, k_s, params)
    E_teo = np.fft.irfft(E_ref_w * H_teo, n=E_sam.size)
    return np.sum((E_sam - E_teo)**2)


ref_file = './data/aux_data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
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

freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
# alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=0, right=alpha_PVA[-1])
# n_PVA = 1.89 * np.ones(f_ref.size)
# alpha_PVA = 52  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
k_PVA = 1e2 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)

freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
n_PLA = np.interp(f_ref, freq_aux, n_PLA, left=n_PLA[0], right=n_PLA[-1])
# alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=alpha_PLA[0], right=alpha_PLA[-1])
alpha_PLA = np.interp(f_ref, freq_aux, alpha_PLA, left=0, right=alpha_PLA[-1])
# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 28  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
k_PLA = 1e2 * TDSC.c_0 * alpha_PLA / (4 * np.pi * f_ref)

n_s = np.array([TDSC.n_air * np.ones(n_PLA.shape), n_PLA, n_PVA, n_PLA, TDSC.n_air * np.ones(n_PLA.shape)])
k_s = np.array([np.zeros(k_PLA.shape), k_PLA, k_PVA, k_PLA, np.zeros(k_PLA.shape)])
# plt.plot(np.zeros(k_PLA.shape))
# plt.plot(k_PLA)
# plt.plot(k_PVA)
# plt.show()


img_dir = './data/img_data/'
file_list = os.listdir(img_dir)
wh = open('./results.txt', 'a')

points = list()

k_bounds = [
    (0e-6, 3e-4),
    (0e-6, 3e-4),
    (0e-6, 3e-4)
]

A_constraint = np.array(
    (
        (1, 1, 1)
        , (1, 0, -1)
    )
)
A_constraint = spy_opt.LinearConstraint(A_constraint, (280e-6, -20e-6), (320e-6, 20e-6))


if __name__ == '__main__':
    for sam_file in tqdm.tqdm(file_list):
        # wh = open('./results.txt', 'a')
        t_sam, E_sam = rd.read_1file(img_dir + sam_file)
        E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
        t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
        sam_file_name = sam_file.split('_')
        posV = sam_file_name[1]
        posH = sam_file_name[3]
        # ref_file = img_dir + 'PosV_25.000000_PosH_' + posH
        ref_file = img_dir + 'PosV_' + str(posV) + '_PosH_25.000000.txt'
        posH = float(posH.replace('.txt', ''))
        posV = float(posV)
        t_ref, E_ref = rd.read_1file(ref_file)
        delta_t_ref = np.mean(np.diff(t_ref))
        E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
        t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
        t_ref *= 1e-12
        f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
        f_ref[0] = 1
        num_statistics = 1
        z1 = list()
        z2 = list()
        z3 = list()
        for num_stats in range(num_statistics):
            res = spy_opt.differential_evolution(cost_function,
                                                 k_bounds,
                                                 args=(E_sam, E_ref_w, f_ref, n_s, k_s),
                                                 # strategy='rand1exp',
                                                 # tol=1e-8,
                                                 # mutation=(0, 1.99),
                                                 # recombination=0.8,
                                                 # popsize=90,
                                                 # maxiter=3000,
                                                 updating='deferred',
                                                 workers=-1,
                                                 disp=False,  # step cost_function value
                                                 polish=True
                                                 # , constraints=A_constraint
                                                 )
            z1.append(res.x[0])
            z2.append(res.x[1])
            z3.append(res.x[2])
        point = (posV, posH, np.mean(z1), np.mean(z2), np.mean(z3))
        wh.write(str(point) + '\n')
