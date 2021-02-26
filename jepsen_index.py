# This script extracts the refractive index of a sample measured in a transmision THz-TDS using the algorithm proposed by
# Jepsen, P.U. J Infrared Milli Terahz Waves (2019) 40: 395. https://doi.org/10.1007/s10762-019-00578-0

# from TDSA import *
from numpy import *
from TDS_constants import *
from DSP_functions import *
# from TDSA import *



def refractive_index(frq, delta_phi, thick, n_environ):
    n = n_environ * ones(delta_phi.size)
    for i in range(delta_phi.size):
        if frq[i] == 0:
            continue
        n[i] += c_0 * delta_phi[i] / (2 * pi * frq[i] * thick)
    return n


def n_quocient(ref_ind):
    return ((ref_ind + n_air) ** 2) / (4 * ref_ind * n_air)


def alpha_w(ref_ind, H_0, thick):
    return - (2 / thick) * log(H_0 * n_quocient(ref_ind))  # m^-1


def jepsen_index(t_ref, E_ref, t_sam, E_sam, thickness, n_env=n_air):
    # Returns refractive index 'n',  absorption coefficient 'alpha_r' and averaged index 'n_avg'

    nSamp = E_ref.size
    nSamp_pow = nextpow2(nSamp)

    # Step 1: Finding the centre of the pulse to get t_0ref and t_0sam
    # pos_t_0ref = centre_loc(E_ref)
    # pos_t_0sam = centre_loc(E_ref)
    pos_t_0ref = centroid_E2(t_ref, E_ref)
    pos_t_0sam = centroid_E2(t_sam, E_sam)
    t_0ref = t_ref[pos_t_0ref]
    t_0sam = t_ref[pos_t_0sam]
    n_avg = 1 + (t_0sam - t_0ref) * c_0 / thickness

    # Step 2: Fourier transform of measures
    f_ref, E_ref_w = fourier_analysis(t_ref, E_ref)  # , nSamp_pow)
    f_sam, E_sam_w = fourier_analysis(t_sam, E_sam)  # , nSamp_pow)
    H_w = E_sam_w / E_ref_w  # complex transfer function

    # Step 3: Calculate reduced phases
    phi_0_ref = 2 * pi * f_ref * t_0ref
    phi_0_sam = 2 * pi * f_sam * t_0sam
    phi_0_ref_red = E_ref_w * exp(- 1j * phi_0_ref)
    phi_0_sam_red = E_sam_w * exp(- 1j * phi_0_sam)
    phi_0_ref_red = angle(phi_0_ref_red)
    phi_0_sam_red = angle(phi_0_sam_red)
    
    # Step 4: Unwrap the reduced phase difference
    delta_phi_0_red = unwrap(phi_0_sam_red - phi_0_ref_red)
    
    # Step 5: Fit the unwrapped phase to a linear function and offset the phase
    f_min_idx, f_max_idx = f_min_max_idx(f_ref, 0.15, 0.35)
    fit_order = 1
    coefs = polyfit(f_ref[f_min_idx:f_max_idx], delta_phi_0_red[f_min_idx:f_max_idx], fit_order)
    
    # Step 6: Undo phase reduction
    delta_phi_0 = delta_phi_0_red - 2 * pi * ones(delta_phi_0_red.size) * round(coefs[fit_order] / (2 * pi), 0)
    delta_phi = abs(delta_phi_0 + (phi_0_sam - phi_0_ref))
    
    # Step 7.1: Obtaining the refractive index
    n = refractive_index(f_ref, delta_phi, thickness, n_env)
    
    T_fk = zeros(n.size)
    T_fk[-1] = (2 * thickness / c_0) * abs(n[0] - n_avg)
    for i in range(n.size - 1):
        T_fk[i] = (2 * thickness / c_0) * abs(n[i] - n_avg + i * (n[i + 1] - n[i]))

    # Step 7.2: Obtaining the absorption coefficient in m^-1
    alpha_f = alpha_w(n, abs(H_w), thickness)

    return n, alpha_f, n_avg


def jepsen_unwrap(t_ref, E_ref, t_sam, E_sam):
    # Returns refractive index 'n',  absorption coefficient 'alpha_r' and averaged index 'n_avg'
    
    nSamp = E_ref.size
    nSamp_pow = nextpow2(nSamp)
    
    # Step 1: Finding the centre of the pulse to get t_0ref and t_0sam
    # pos_t_0ref = centre_loc(E_ref)
    # pos_t_0sam = centre_loc(E_ref)
    pos_t_0ref = centroid_E2(t_ref, E_ref)
    pos_t_0sam = centroid_E2(t_sam, E_sam)
    t_0ref = t_ref[pos_t_0ref]
    t_0sam = t_ref[pos_t_0sam]
    
    # Step 2: Fourier transform of measures
    f_ref, E_ref_w = fourier_analysis(t_ref, E_ref)  # , nSamp_pow)
    f_sam, E_sam_w = fourier_analysis(t_sam, E_sam)  # , nSamp_pow)
    H_w = E_sam_w / E_ref_w  # complex transfer function
    
    # Step 3: Calculate reduced phases
    phi_0_ref = 2 * pi * f_ref * t_0ref
    phi_0_sam = 2 * pi * f_sam * t_0sam
    phi_0_ref_red = E_ref_w * exp(- 1j * phi_0_ref)
    phi_0_sam_red = E_sam_w * exp(- 1j * phi_0_sam)
    phi_0_ref_red = angle(phi_0_ref_red)
    phi_0_sam_red = angle(phi_0_sam_red)
    
    # Step 4: Unwrap the reduced phase difference
    delta_phi_0_red = unwrap(phi_0_sam_red - phi_0_ref_red)
    
    # Step 5: Fit the unwrapped phase to a linear function and offset the phase
    f_min_idx, f_max_idx = f_min_max_idx(f_ref, 0.15, 0.35)
    fit_order = 1
    coefs = polyfit(f_ref[f_min_idx:f_max_idx], delta_phi_0_red[f_min_idx:f_max_idx], fit_order)
    
    # Step 6: Undo phase reduction
    delta_phi_0 = delta_phi_0_red - 2 * pi * ones(delta_phi_0_red.size) * round(coefs[fit_order] / (2 * pi), 0)
    delta_phi = abs(delta_phi_0 + (phi_0_sam - phi_0_ref))
    
    return delta_phi
