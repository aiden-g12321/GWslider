from GW_class import *
from template import waveform
import constants as c
import numpy as np
from scipy.signal.windows import tukey
from template import *


def new_matched_filter(template, data, GWsignal, det):
    """Runs the matched filter calculation given a specific real template, strain
    data, time, psd, and sample rate. Finds the offset and phase to maximize
    SNR, as well as effective distance and horizon

    Args:
        template (ndarray): real part of initial template corresponding to event
        data (ndarray): strain data near event
        time (ndarray): time near event
        data_psd (interpolating function): psd of strain data around event
        fs (float): sample rate of data

    Returns:
        float: maximum SNR value obtained
        float: time of maximum SNR value
        float: effective distance found
        float: horizon found
        float: template phase which maximizes SNR
        float: template offset which maximizes SNR
    """

    # get frequencies of our data
    datafreq= c.freqs
    # spacing between frequency bins
    df = c.freqs[1] - c.freqs[0]

    # get psd values from data dictionary
    psd_func = GWsignal['large_data_psds'][det]
    psd = psd_func(datafreq)

    # calculate inner product for normalization
    integrand = np.real(template * template.conjugate()) / psd
    inner_product = 4 * np.sum(integrand) * df

    # normalization 
    p_normalization_factor = np.sqrt(inner_product)
    #normalized template
    p = template / p_normalization_factor

    # time shifts to search over for optimization
    time_shifts = waveform.times_full - waveform.times_full[0]

    # for taking the fft of our template and data
    dwindow= tukey(data.size, alpha= 1./4)
    data_FD = np.fft.rfft(data*dwindow) * dt
    
    # time shifts in frequency-domain (complex phase)
    time_shift_in_FD = np.exp(2. * np.pi * 1.j * c.freqs[:, None] * time_shifts[None, :])
    integrand = (data_FD * p.conjugate() / psd)[:, None] * time_shift_in_FD
    
    # do Fourier transform manually
    z = 4. * np.sum(integrand, axis=0) * df 


    # # shift the SNR vector by the template length so that the peak is at
    # # the end of the template
    # peaksample = int(data.size / 2)  # location of peak in the template
    # SNR_complex = np.roll(z, peaksample)
    # SNRmax= abs(SNR_complex)


    # get optimal time-shift, phase, and amplitude
    opt_ndx = np.argmax(np.abs(z))
    opt_time_shift = time_shifts[opt_ndx]
    opt_amplitude = np.abs(z)[opt_ndx]
    opt_phase = np.angle(z[opt_ndx])
    SNRmax = np.abs(z)[opt_ndx]
    

    return SNRmax, opt_time_shift, opt_amplitude, opt_phase, p


def opt_template(template, GWsignal, det):
    # template should be normalized template p

    # import GWsignal dictionary
    time = GWsignal['time']
    time_center = GWsignal['time_center']
    fs = GWsignal['fs']
    

    # amount of data we want to calculate matched filter SNR over- up to 32s
    data_time_window = time[len(time) - 1] - time[0] - (32 - 4)

    time_filter_window = np.where((time <= time_center + data_time_window * .5) & 
                                (time >= time_center - data_time_window * .5))
    time_filtered = time[time_filter_window]
    
    # get data 
    strain = GWsignal[det]['strain'][time_filter_window]
    strain_whitenbp = GWsignal[det]['strain_whitenbp'][time_filter_window]


    SNRmax, opt_time_shift, opt_amplitude, opt_phase, p = new_matched_filter(template, strain, GWsignal, det)

    # get frequencies of our data
    datafreq= c.freqs

    # get psd values from data dictionary
    psd_func = GWsignal['large_data_psds'][det]
    psd = psd_func(datafreq)

    # optimal template in frequency-domain
    # scale by optimal frequency
    opt_template_FD = opt_amplitude *p
    # optimal time shift
    opt_template_FD *= np.exp(-2. * np.pi * 1.j * c.freqs * opt_time_shift)
    # add phase shift
    opt_template_FD *= np.exp(1.j * opt_phase)

    # whiten template
    opt_template_FD_whitened = opt_template_FD / np.sqrt(psd)

    # get whitened template in TD 
    opt_template_TD = np.fft.irfft(opt_template_FD_whitened, waveform.times_full.shape[0]) *4*np.sqrt(fs)

    return  opt_template_TD, strain_whitenbp, time_filtered - time_center, SNRmax, opt_amplitude, opt_phase



# # wrapper function for matched filter
def wrapped_matched_filter(params, GW_signal, det):
     return opt_template(get_template(params, GW_signal.dictionary), GW_signal.dictionary, det)

# def residual_func(data, fit):
#     return data-fit