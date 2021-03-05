import numpy as np
import jepsen_index as jpidx
import read_data as rd
import DSP_functions as DSPf
import matplotlib.pyplot as plt
import aux_functions as axf
import H_T3
import TDS_constants as TDSC
import scipy.optimize as spy_opt
import tqdm


for i in range(3):
    t_ref, E_ref = rd.read_1file('./PVA/ref' + str(i+1) + '.txt')
    t_sam, E_sam = rd.read_1file('./PVA/sam' + str(i + 1) + '.txt')
    t_ref *= 1e-12
    t_sam *= 1e-12
    f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
    n, alpha, n_avg = jpidx.jepsen_index(t_ref, E_ref, t_sam, E_sam, 200e-6)
    plt.figure(1)
    plt.plot(f_ref, n)
    plt.figure(2)
    plt.plot(f_ref, alpha*1e-2)


for i in range(3):
    t_ref, E_ref = rd.read_1file('./PLA/ref' + str(i+1) + '.txt')
    t_sam, E_sam = rd.read_1file('./PLA/sam' + str(i + 1) + '.txt')
    t_ref *= 1e-12
    t_sam *= 1e-12
    f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
    n, alpha, n_avg = jpidx.jepsen_index(t_ref, E_ref, t_sam, E_sam, 140e-6)
    plt.figure(3)
    plt.plot(f_ref, n)
    plt.figure(4)
    plt.plot(f_ref, alpha*1e-2)


plt.show()
