import DSP_functions as DSPf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scist


times = np.arange(1, 5001) * 1e-2  # N = 5000, 1.00 to 50.00
# times *= 1e-12  # ps
pulse = np.cos(np.pi * times + 2 * np.pi * np.random.rand()) * scist.norm.pdf(times, loc=10, scale=1.2)
f_ref, E_ref_w = DSPf.fourier_analysis(times, pulse)
# plt.plot(f_ref, 20*np.log10(E_ref_w))
plt.plot(times, pulse)
plt.show()
