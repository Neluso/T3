import numpy as np
import H_T3
import TDS_constants as TDSC
import read_data as rd
import os
import matplotlib.pyplot as plt
import DSP_functions as DSPf
import scipy.optimize as spy_opt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(params, *args):
    d_air, thick_1, thick_2, thick_3 = params
    E_sam, E_ref_w, freqs, n_PVA, k_PVA, n_PLA, k_PLA = args
    H1 = H_T3.H_sim(freqs, n_PLA, k_PLA, thick_1, TDSC.n_air, 0, n_PVA, k_PVA)
    H2 = H_T3.H_sim(freqs, n_PVA, k_PVA, thick_2, n_PLA, k_PLA, n_PLA, k_PLA)
    H3 = H_T3.H_sim(freqs, n_PLA, k_PLA, thick_3, n_PVA, k_PVA, TDSC.n_air, 0)
    E_teo = - np.fft.irfft(H1 * H2 * H3 * E_ref_w)
    return sum((E_sam - E_teo) ** 2)


ref_file = './data/aux_data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
t_ref = - np.flip(t_ref)
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref[0] = 1
n_PVA = 1.8 * np.ones(f_ref.size)
alpha_PVA = 40  # cm^-1 / THz
alpha_PVA = 40 * f_ref * 1e-12
k_PVA = 1e-10 * TDSC.c_0 * alpha_PVA / (4 * np.pi * f_ref)
n_PLA = 1.6 * np.ones(f_ref.size)
alpha_PLA = 35  # cm^-1 / THz
alpha_PLA = 35 * f_ref * 1e-12
k_PLA = 35  # cm^-1 / THz


img_dir = './data/img_data/'
file_list = os.listdir(img_dir)

points = list()

k_bounds = [(-1e-3, 1e-3),  # d_air
            # (1, 2),  # n
            # (0, 1),  # k
            (1e-5, 5e-4),  # thickness
            # (1, 2),  # n
            # (0, 1),  # k
            (1e-5, 5e-4),  # thickness
            # (1, 2),  # n
            # (0, 1),  # k
            (1e-5, 5e-4)  # thickness
            ]

x = list()
y = list()
zs1 = list()
zs2 = list()
zs3 = list()

if __name__ == '__main__':
    for sam_file in file_list:
        t_sam, E_sam = rd.read_1file(img_dir + sam_file)
        t_sam = - np.flip(t_sam)
        t_sam *= 1e-12
        sam_file_name = sam_file.split('_')
        posV = float(sam_file_name[1])
        posH = float(sam_file_name[3].replace('.txt', ''))
        print(str(posV) + '/55.5', str(posH) + '/39.75')
        num_statistics = 10
        z1 = list()
        z2 = list()
        z3 = list()
        for num_stats in range(num_statistics):
            res = spy_opt.differential_evolution(cost_function,
                                                 k_bounds,
                                                 args=(E_sam, E_ref_w, f_ref, n_PVA, k_PVA, n_PLA, k_PLA),
                                                 # popsize=30,
                                                 # maxiter=3000,
                                                 updating='deferred',
                                                 workers=-1,
                                                 disp=False,  # step cost_function value
                                                 polish=True
                                                 )
            z1.append(res.x[1])
            z3.append(res.x[2])
            z3.append(res.x[3])
        points.append((posV, posH, np.mean(z1), np.mean(z2), np.mean(z3)))
        # print((posV, posH, res.x[1], res.x[2], res.x[3]))
        x.append(posH)
        y.append(posV)
        zs1.append(res.x[1])
        zs2.append(res.x[2])
        zs3.append(res.x[3])

    wh = open('./results.txt', 'a')
    for point in points:
        wh.write(str(point) + '\n')
