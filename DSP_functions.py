from numpy.fft import *
from numpy.fft import fft as fft_func
from numpy import *
from scipy import signal
from aux_functions import *
import pywt


def fourier_analysis(t_data, E_data, nSamp=0):
    if nSamp == 0:
        nSamp = E_data.size
    # samp_int = t_data[1] - t_data[0]  # seconds
    samp_int = float(mean(diff(t_data)))  # seconds
    E_data_w = rfft(E_data, n=nSamp)
    f_data = rfftfreq(nSamp, d=samp_int)  # Hz
    # E_data_w = fft_func(E_data, n=nSamp)
    # f_data = fftfreq(nSamp, d=samp_int)  # Hz
    # f_data = arange(nSamp) / nSamp
    # f_data /= samp_int
    return f_data, E_data_w


def fourier_analysis_comp(t_data, E_data, nSamp=0):
    if nSamp == 0:
        nSamp = E_data.size
    samp_int = float(mean(diff(t_data)))  # seconds
    E_data_w = fft_func(E_data, n=nSamp)
    f_data = fftfreq(nSamp, d=samp_int)  # Hz
    return f_data, E_data_w


def centre_loc(E_data):  # finds the pulse centre  based on pulse maximum of absolute signal TODO improve algorithm
    t_0_pos = argmax(abs(E_data))
    return t_0_pos


def centroid_E2(t_val, E_val):  # t_centroid
    t_centroid = sum(t_val * abs(E_val)**2) / sum(abs(E_val)**2)
    t_idx = where(t_val <= t_centroid)[0]
    return t_idx[-1]


def noise_floor(freq, E_data, f_lim):
    f_lim_idx = where(freq >= f_lim)[0][0]
    p = polyfit(freq[f_lim_idx:], E_data[f_lim_idx:], 1)
    return p[1] * ones(freq.size)


def zero_padding(val, n0_l, n0_r):
    if n0_l > 0:
        val = concatenate((zeros(n0_l), val))
    if n0_r > 0:
        val = concatenate((val, zeros(n0_r)))
    return val


def force_exp_func(data_size, top_half_size, decay_length, decay_minimum=1):  # decay_minimum default is 1%
    decay_rate = - log(decay_minimum / 100) / decay_length
    x = linspace(0, decay_length, num=decay_length)
    decay = exp(- x * decay_rate)
    half_point = int(data_size / 2)
    y = zeros(data_size)
    for i in range(y.size):
        if half_point - top_half_size <= i < half_point + top_half_size:
            y[i] = 1
        if half_point + top_half_size <= i < decay.size + half_point + top_half_size - 1:
            y[i] = decay[i - half_point - top_half_size]
    return y


def force_exp_windowing(t_val, E_val, t_sub, E_sub):
    center_val_idx = centre_loc(E_val)
    center_sub_idx = centre_loc(E_sub)
    left_padd_idxs = E_val.size - 2 * center_val_idx
    left_padd_idxs_sub = E_sub.size - 2 * center_sub_idx
    E_val = zero_padding(E_val, left_padd_idxs, 0)
    E_sub = zero_padding(E_sub, left_padd_idxs_sub, 0)
    t_val_rev = - flip(t_val[1:left_padd_idxs + 1])
    t_val = concatenate((t_val_rev, t_val))
    t_sub = concatenate((t_val_rev, t_sub))
    E_val *= force_exp_func(E_val.size, int(center_val_idx / 2), int(center_val_idx / 4))
    E_sub *= force_exp_func(E_sub.size, int(center_sub_idx / 2), int(center_sub_idx / 4))
    E_sub = zero_padding(E_sub, left_padd_idxs - left_padd_idxs_sub, 0)
    return t_val, E_val, t_sub, E_sub


def bh_windowing(t_val, E_val, t_sub, E_sub):
    center_val_idx = centre_loc(E_val)
    center_sub_idx = centre_loc(E_sub)
    left_padd_idxs = E_val.size - 2 * center_val_idx
    left_padd_idxs_sub = E_sub.size - 2 * center_sub_idx
    E_val = zero_padding(E_val, left_padd_idxs, 0)
    E_sub = zero_padding(E_sub, left_padd_idxs_sub, 0)
    t_val_rev = - flip(t_val[1:left_padd_idxs + 1])
    t_val = concatenate((t_val_rev, t_val))
    t_sub = concatenate((t_val_rev, t_sub))
    E_val *= signal.windows.blackmanharris(E_val.size)
    E_sub *= signal.windows.blackmanharris(E_sub.size)
    E_sub = zero_padding(E_sub, left_padd_idxs - left_padd_idxs_sub, 0)
    return t_val, E_val, t_sub, E_sub


def cheb_windowing(t_val, E_val, t_sub, E_sub):
    center_val_idx = centre_loc(E_val)
    center_sub_idx = centre_loc(E_sub)
    left_padd_idxs = E_val.size - 2 * center_val_idx
    left_padd_idxs_sub = E_sub.size - 2 * center_sub_idx
    E_val = zero_padding(E_val, left_padd_idxs, 0)
    E_sub = zero_padding(E_sub, left_padd_idxs_sub, 0)
    t_val_rev = - flip(t_val[1:left_padd_idxs + 1])
    t_val = concatenate((t_val_rev, t_val))
    t_sub = concatenate((t_val_rev, t_sub))
    E_val *= signal.windows.chebwin(E_val.size, 80)
    E_sub *= signal.windows.chebwin(E_sub.size, 80)
    E_sub = zero_padding(E_sub, left_padd_idxs - left_padd_idxs_sub, 0)
    return t_val, E_val, t_sub, E_sub


def hann_windowing(t_val, E_val, t_sub, E_sub):
    center_val_idx = centre_loc(E_val)
    center_sub_idx = centre_loc(E_sub)
    left_padd_idxs = E_val.size - 2 * center_val_idx
    left_padd_idxs_sub = E_sub.size - 2 * center_sub_idx
    E_val = zero_padding(E_val, left_padd_idxs, 0)
    E_sub = zero_padding(E_sub, left_padd_idxs_sub, 0)
    t_val_rev = - flip(t_val[1:left_padd_idxs + 1])
    t_val = concatenate((t_val_rev, t_val))
    t_sub = concatenate((t_val_rev, t_sub))
    E_val *= signal.windows.hann(E_val.size)
    E_sub *= signal.windows.hann(E_sub.size)
    E_sub = zero_padding(E_sub, left_padd_idxs - left_padd_idxs_sub, 0)
    return t_val, E_val, t_sub, E_sub


def tukey_windowing(t_val, E_val, t_sub, E_sub):
    center_val_idx = centre_loc(E_val)
    center_sub_idx = centre_loc(E_sub)
    left_padd_idxs = E_val.size - 2 * center_val_idx
    left_padd_idxs_sub = E_sub.size - 2 * center_sub_idx
    E_val = zero_padding(E_val, left_padd_idxs, 0)
    E_sub = zero_padding(E_sub, left_padd_idxs_sub, 0)
    t_val_rev = - flip(t_val[1:left_padd_idxs + 1])
    t_val = concatenate((t_val_rev, t_val))
    t_sub = concatenate((t_val_rev, t_sub))
    E_val *= signal.windows.tukey(E_val.size)
    E_sub *= signal.windows.tukey(E_sub.size)
    E_sub = zero_padding(E_sub, left_padd_idxs - left_padd_idxs_sub, 0)
    return t_val, E_val, t_sub, E_sub


def rect_low_filter(E_val_w, signal_percent):
    band_lim_idx = int(round(E_val_w.size * signal_percent))
    pass_band = ones(band_lim_idx)
    att_band = zeros(E_val_w.size - band_lim_idx)
    filt = concatenate((pass_band, att_band))
    return E_val_w * filt


def lin_low_filter(E_val_w, signal_percent, slope):  # todo: slope = dB/octave
    band_lim_idx = int(round(E_val_w.size * signal_percent))
    pass_band = ones(band_lim_idx)
    att_band = arange(E_val_w.size - band_lim_idx) / (E_val_w.size - band_lim_idx)
    filt = concatenate((pass_band, flip(att_band)))
    return E_val_w * filt


def gauss_low_filter(f_val, cutoff, sigma):
    sigma_idx = int(round(sigma / mean(diff(f_val))))
    cutoff_idx = where(f_val >= cutoff)[0][0]
    low_band = ones(cutoff_idx)
    upper_band = signal.windows.gaussian(2 * (f_val.size - low_band.size), std=sigma_idx)
    pass_band = concatenate((low_band, upper_band[int(round(upper_band.size / 2)):]))
    return pass_band


def wiener_filter(E_data, beta=0.00003162277):  # beta = 1/SNR
    return E_data*conj(E_data) / (E_data*conj(E_data) + beta)


def SWT_denoising(E_data, level, threshold):
    sym4_coeffs = pywt.swt(E_data, 'sym4', level)
    threshold *= sqrt(2 * log(E_data.size))
    filtered_coeffs = list()
    for pair in sym4_coeffs:
        cA_mod = pair[0]
        cD_mod = pair[1]
        cD_aux = zeros(cD_mod.shape)
        for i in range(cD_mod.size):
            if cD_mod[i] >= threshold:
                cD_aux[i] = cD_mod[i] - threshold
            elif cD_mod[i] <= - threshold:
                cD_aux[i] = cD_mod[i] + threshold
            else:
                cD_aux[i] = 0
        filtered_coeffs.append((cA_mod, cD_aux))

    return pywt.iswt(array(filtered_coeffs), 'sym4')
