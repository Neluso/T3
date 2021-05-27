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


ref_file = './data/aux_data/ref.txt'
t_ref, E_ref = rd.read_1file(ref_file)
t_ref *= 1e-12
f_ref, E_ref_w = DSPf.fourier_analysis(t_ref, E_ref)
f_ref[0] = 1

img_dir = './data/img_data/'
file_list = os.listdir(img_dir)

pixels = list()

for sam_file in file_list:
    t_sam, E_sam = rd.read_1file(img_dir + sam_file)
    sam_file_name = sam_file.split('_')
    posV = float(sam_file_name[1])
    posH = float(sam_file_name[3].replace('.txt', ''))
    intensity = sum(E_sam**2)
    pixels.append((posH, posV, intensity))

pixels = np.array(pixels)
print(pixels[:, 0])
print(pixels[:, 1])
# print(pixels)
# quit()
plt.imshow(pixels, origin="upper",interpolation="nearest", aspect='equal')
# plt.colorbar()
plt.show()