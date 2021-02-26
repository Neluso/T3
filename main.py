import numpy as np
import H_T3
import TDS_constants as TDSC
import read_data as rd
import os
import matplotlib.pyplot as plt
import DSP_functions as DSPf
import scipy.optimize as spy_opt


def cost_function(params, *args):
    d_air, n_1, k_1, thick_1, n_2, k_2, thick_2, n_3, k_3, thick_3 = params
    E_sam, E_ref_w, freqs = args
    H1 = H_T3.H_sim(freqs, n_1, k_1, thick_1, TDSC.n_air, 0, n_2, k_2)
    H2 = H_T3.H_sim(freqs, n_2, k_2, thick_2, n_1, k_1, n_3, k_3)
    H3 = H_T3.H_sim(freqs, n_3, k_3, thick_3, n_2, k_2, TDSC.n_air, 0)
    E_teo = - np.fft.irfft(H1 * H2 * H3 * E_ref_w)
    return sum((E_sam - E_teo) ** 2)


ref_file = './data/aux_data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
t_ref = - np.flip(t_ref)
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)

img_dir = './data/img_data/'
file_list = os.listdir(img_dir)

points = list()

k_bounds = [(-1e-3, 1e-3),  # d_air
            (1, 2),  # n
            (0, 1),  # k
            (1e-5, 5e-4),  # thickness
            (1, 2),  # n
            (0, 1),  # k
            (1e-5, 5e-4),  # thickness
            (1, 2),  # n
            (0, 1),  # k
            (1e-5, 5e-4)  # thickness
            ]

if __name__ == '__main__':
    for sam_file in file_list:
        t_sam, E_sam = rd.read_1file(img_dir + sam_file)
        t_sam = - np.flip(t_sam)
        t_sam *= 1e-12
        sam_file_name = sam_file.split('_')
        posV = float(sam_file_name[1])
        posH = float(sam_file_name[3].replace('.txt', ''))
        print(posV, posH)

        res = spy_opt.differential_evolution(cost_function,
                                             k_bounds,
                                             args=(E_sam, E_ref_w, f_ref),
                                             popsize=30,
                                             maxiter=3000,
                                             updating='deferred',
                                             workers=-1,
                                             disp=False,  # step cost_function value
                                             polish=True
                                             )
        points.append((posV, posH, res.x[3], res.x[6], res.x[9]))
        print((posV, posH, res.x[3], res.x[6], res.x[9]))
    # print(points)
