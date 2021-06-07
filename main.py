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
    E_sam, E_ref_w, freqs, n_PVA, k_PVA, n_PLA, k_PLA = args
    # thick_1, thick_2, thick_3, d_air = params
    thick_1, thick_2, thick_3 = params
    # thick_1, thick_2, thick_3, nv, kv, nl, kl = params
    # thick_1, thick_2, thick_3, delta_1, delta_2, delta_3, delta_4 = params
    # thick, n, k = params

    # print(params)

    # thick_1 = param_2_thickess(thick_1)
    # thick_2 = param_2_thickess(thick_2)
    # thick_3 = param_2_thickess(thick_3)
    # n_PVA *= delta_1
    # k_PVA *= delta_2
    # n_PLA *= delta_3
    # k_PLA *= delta_4

    # n_PVA = nv
    # k_PVA = kv * freqs * 1e-12
    # k_PVA[0] = 0
    # n_PLA = nl
    # k_PLA = kl * freqs * 1e-12
    # k_PLA[0] = 0

    n_s = [TDSC.n_air, n_PLA, n_PVA, n_PLA, TDSC.n_air]
    k_s = [0, k_PLA, k_PVA, k_PLA, 0]
    thick_s = [thick_1, thick_2, thick_3]

    # thick_s = [thick]
    # n_s = [TDSC.n_air, n, TDSC.n_air]
    # k_s = [0, k * f_ref * 1e-12, 0]

    H_teo = H_T3.H_sim_rouard(freqs, n_s, k_s, thick_s)
    # w_filt = DSPf.wiener_filter(E_ref_w)
    # H_teo *= w_filt
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

# n_PVA = 1.83 * np.ones(f_ref.size)
# alpha_PVA = 48  # cm^-1 / THz
# alpha_PVA = alpha_PVA * f_ref * 1e-12
freq_aux, n_PVA, n_PVA_std, alpha_PVA, alpha_PVA_std = rd.read_from_1file('./PLA_PVA/PVA.txt')
n_PVA = np.interp(f_ref, freq_aux, n_PVA, left=n_PVA[0], right=n_PVA[-1])
alpha_PVA = np.interp(f_ref, freq_aux, alpha_PVA, left=alpha_PVA[0], right=alpha_PVA[-1])
k_PVA = 1e-10 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)

# n_PLA = 1.62 * np.ones(f_ref.size)
# alpha_PLA = 32  # cm^-1 / THz
# alpha_PLA = alpha_PLA * f_ref * 1e-12
freq_aux, n_PLA, n_PLA_std, alpha_PLA, alpha_PLA_std = rd.read_from_1file('./PLA_PVA/PLA.txt')
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

img_dir = './data/img_data/'
file_list = os.listdir(img_dir)
wh = open('./results.txt', 'a')

points = list()

# thick_tupl = (20e-6, 450e-6)
thick_tupl = (0, 350e-6)
correct_tuple = (0.9, 1.1)

k_bounds = [
    (20e-6, 135e-6),
    (40e-6, 205e-6),
    (20e-6, 135e-6)
    # , (- 1000e-6, 0)
    # , (correct_tuple)
    # , correct_tuple
    # , correct_tuple
    # , correct_tuple
]

A_constraint = np.array(
    (
        (1, 1, 1)  # , 0, 0, 0, 0),
        # (1, 0, -1)  # , 0, 0, 0, 0)
    )
)

x = list()
y = list()
zs1 = list()
zs2 = list()
zs3 = list()


if __name__ == '__main__':
    for sam_file in tqdm.tqdm(file_list):
        # wh = open('./results.txt', 'a')
        t_sam, E_sam = rd.read_1file(img_dir + sam_file)
        E_sam = DSPf.zero_padding(E_sam, 0, enlargement)
        t_sam = np.concatenate((t_sam, t_sam[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))

        plt.plot(t_sam, E_sam)
        # t_sam = - np.flip(t_sam)
        t_sam *= 1e-12
        sam_file_name = sam_file.split('_')
        posV = float(sam_file_name[1])
        posH = sam_file_name[3]
        ref_file = img_dir + 'PosV_25.000000_PosH_' + posH
        posH = float(posH.replace('.txt', ''))
        t_ref, E_ref = rd.read_1file(ref_file)
        delta_t_ref = np.mean(np.diff(t_ref))
        E_ref = DSPf.zero_padding(E_ref, 0, enlargement)
        t_ref = np.concatenate((t_ref, t_ref[-1] * np.ones(enlargement) + delta_t_ref * np.arange(1, enlargement + 1)))
        t_ref *= 1e-12
        f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
        f_ref[0] = 1
        # if 44.0 < posV < 46.0:
        #     if 25.0 < posH < 30.0:
        #         print(posV, posH)
        #         break
        # print(str(posV) + '/77.0', str(posH) + '/10.0')
        num_statistics = 1
        z1 = list()
        z2 = list()
        z3 = list()
        for num_stats in range(num_statistics):
            # if 72.5 <= posV <= 77.2:
            #     break
            # elif 5.0 <= posH <= 10.0:
            #     break
            res = spy_opt.differential_evolution(cost_function,
                                                 k_bounds,
                                                 args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
                                                 # strategy='rand1exp',
                                                 # tol=1e-8,
                                                 # mutation=(0, 1.99),
                                                 # recombination=0.4,
                                                 # bounds=k_bounds,
                                                 # popsize=30,
                                                 # maxiter=3000,
                                                 updating='deferred',
                                                 workers=-1,
                                                 disp=False,  # step cost_function value
                                                 polish=True
                                                 , constraints=spy_opt.LinearConstraint(A_constraint, (265e-6),  # , -20e-6),
                                                                                        (335e-6))  # , 20e-6))
                                                 )
            z1.append(res.x[0])
            z2.append(res.x[1])
            z3.append(res.x[2])
            # z1.append(param_2_thickess(res.x[0]))
            # z2.append(param_2_thickess(res.x[2]))
            # z3.append(param_2_thickess(res.x[2]))
        # H1 = H_T3.H_sim(f_ref, n_PLA, k_PLA, np.mean(z1), TDSC.n_air, 0, n_PVA, k_PVA)
        # H2 = H_T3.H_sim(f_ref, n_PVA, k_PVA, np.mean(z2), n_PLA, k_PLA, n_PLA, k_PLA)
        # H3 = H_T3.H_sim(f_ref, n_PLA, k_PLA, np.mean(z3), n_PVA, k_PVA, TDSC.n_air, 0)
        # air_phase = H_T3.phase_factor(TDSC.n_air, 0, - (np.mean(z1) + np.mean(z2) + np.mean(z3)), f_ref)
        # E_teo = np.fft.irfft(H1 * H2 * H3 * E_ref_w)
        # plt.plot(t_ref * 1e12, E_teo)
        # print(np.mean(z1)*1e6, np.mean(z2)*1e6, np.mean(z3)*1e6)
        # plt.show()
        point = (posV, posH, np.mean(z1), np.mean(z2), np.mean(z3))
        # point = (posV, posH, z1[0], z2[0], z3[0])
        # points.append(point)
        # print(point)
        wh.write(str(point) + '\n')
        # wh.close()
        # print((posV, posH, res.x[1], res.x[2], res.x[3]))
        # x.append(posH)
        # y.append(posV)
        # zs1.append(res.x[1])
        # zs2.append(res.x[2])
        # zs3.append(res.x[3])
