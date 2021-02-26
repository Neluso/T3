import numpy as np
import jepsen_index as jpidx
import read_data as rd
import DSP_functions as DSPf
import matplotlib.pyplot as plt
import H_T3


for i in range(3):
    t_ref, E_ref = rd.read_1file('./PVA/ref' + str(i+1) + '.txt')
    t_sam, E_sam = rd.read_1file('./PVA/sam' + str(i + 1) + '.txt')
    t_ref *= 1e-12
    t_sam *= 1e-12
    f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
    n, alpha, n_avg = jpidx.jepsen_index(t_ref, E_ref, t_sam, E_sam, 200e-6)
    plt.plot(f_ref, n)
for i in range(3):
    t_ref, E_ref = rd.read_1file('./PLA/ref' + str(i+1) + '.txt')
    t_sam, E_sam = rd.read_1file('./PLA/sam' + str(i + 1) + '.txt')
    t_ref *= 1e-12
    t_sam *= 1e-12
    f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
    n, alpha, n_avg = jpidx.jepsen_index(t_ref, E_ref, t_sam, E_sam, 140e-6)
    plt.plot(f_ref, n)
plt.show()
