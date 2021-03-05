from numpy import where  # array functions
from numpy import log10  # mathematical functions
from time import time_ns, strftime, gmtime


def f_min_max_idx(freq, fmin=0.1, fmax=1):
    f_min_idx = 1
    f_max_idx = 100
    f_min = freq[f_min_idx]
    f_max = freq[f_max_idx]
    for frq in freq:
        if frq <= fmin * 1e12:  # 0.1 THz default
            f_min = frq
        if frq <= fmax * 1e12:  # 1 THz default
            f_max = frq
    f_min_idx = where(freq == f_min)[0][0]
    f_max_idx = where(freq == f_max)[0][0]
    return f_min_idx, f_max_idx


def toDb(x):
    return 20 * log10(abs(x))


def toDb_0(x):
    return 20 * log10(abs(x)/max(abs(x)))


def fromDb(x):
    return 10**(x / 20)


def prettyfy(x, norm):
    return toDb(x / norm)


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def f_range(freq, f_lim_min, f_lim_max):
    for frq in freq:
        if frq <= f_lim_min:
            f_min = frq
        if frq <= f_lim_max:
            f_max = frq
    f_min_idx = where(freq == f_min)[0][0]
    f_max_idx = where(freq == f_max)[0][0]
    return f_min_idx, f_max_idx


def print_time_ns(t1, t2):
    secs = (t2 - t1) * 1e-9
    if secs < 3600:
        print('Processing time (mm:ss):', strftime('%M:%S', gmtime(secs)))
    else:
        print('Processing time (mm:ss):', strftime('%H:%M:%S', gmtime(secs)))
    return 0
