'''Use matched filtering technique to find optimal amplitude and phase for template.'''


import numpy as np
from scipy.signal.windows import tukey
from signal_processing import whiten, bandpass
import matplotlib.pyplot as plt
from widgets import *
from template import get_template
import constants as c
from new_matched_filter import *

def matched_filter(template, data, time, data_psd, fs):
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
    # get the fourier frequencies of data for when we fft (for dividing by psd)


    #datafreq = np.fft.fftfreq(template.size)*fs
    datafreq= c.freqs  # multiply by fs?
    #df = np.abs(datafreq[1] - datafreq[0])
    # for taking the fft of our template and data
    #dwindow = tukey(template.size, alpha=1./4)
    dwindow= tukey(data.size, alpha= 1./4)

    # compute the template and data ffts.

     #ADDED multply by dt instead of dividing by fs
    template = template / fs
    #template_fft = np.fft.fft(template*dwindow) / fs
    data_fft = np.fft.rfft(data*dwindow) /fs
    
    # use the larger psd of the data calculated earlier for a better calculation
    # power_vec = list(map(data_psd, np.abs(datafreq)))
    power_vec = data_psd(np.abs(datafreq))
    print(data_psd(freqs).min(), data_psd(freqs).max())
    # -- Zero out negative frequencies
    #negindx = np.where(datafreq<0)
    #data_fft[negindx] = 0

    # -- Calculate the matched filter output in the time domain: Multiply
    # the Fourier Space template and data, and divide by the noise power in
    # each frequency bin.  Taking the Inverse Fourier Transform (IFFT) of
    # the filter output puts it back in the time domain, so the result will
    # be plotted as a function of time off-set between the template and the
    # data:
    optimal = data_fft * template.conjugate() / power_vec
    optimal_time = 4 * np.fft.irfft(optimal) * fs

    
    # -- Normalize the matched filter output: Normalize the matched filter
    # output so that we expect an average value of 1 at times of just noise.  Then,
    # the peak of the matched filter output will tell us the
    # signal-to-noise ratio (SNR) of the signal.
    ## multiply by 4 instead of 2  
    sigmasq = 4 * (template* template.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    # shift the SNR vector by the template length so that the peak is at
    # the end of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]
    
    print('sigma', sigma)
    print('SNRmax', SNRmax)
    # Calculate the effective distance
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distance
    horizon = sigma/8

    # Extract time offset and phase at peak
    phase = -np.angle(SNR_complex[indmax])
    offset = (indmax-peaksample)

    return SNRmax, timemax, d_eff, horizon, phase, offset




def get_shifted_data(template, fband, filter_data, data_psd, dt):
    #first variable was tempalte_p
    """Obtains data shifts of templates and residual data after having found the
    best fit phase and offsets for the template.

    Args:
        template_p (ndarray): real (plus-polarization) part of template
        strain (ndarray): strain data
        time (ndarray): time
        strain_whiten (ndarray): whitened strain data
        strain_whitenbp (ndarray): whitened and bandpassed strain data
        fband (list): low and high pass filters for the template bandpass
        filter_data (dict): dictionary containing phase, offset, d_eff values
            for given matched filter calculation
        data_psd (interpolating function): function which outputs a power value
            for a given data frequency

    Returns:
        ndarray: whitened, bandpassed, phaseshifted and offset template
        ndarray: phaseshifted and offset residual data
        ndarray: whitened, phaseshifted and offset residual data
        ndarray: whitened, bandpassed, phaseshifted and offset residual data
    """
    d_eff = filter_data['d_eff']
    phase = filter_data['phase']
    offset = filter_data['offset']
    
    # whiten and bandpass template_p for plotting- also applying phase shift,
    # amplitude scale
    template_whitened = whiten(template / d_eff, data_psd, dt,
                               phase_shift=phase, time_shift=(offset * dt))
    template_match = bandpass(template_whitened, fband, 1. / dt)

    hf = np.fft.rfft(template_whitened)
    freqs = np.fft.rfftfreq(len(template_whitened), dt)
    df = freqs[1] - freqs[0]

    # Compute (h|h)
    sigmasq = 4 * np.sum((hf * hf.conjugate()) / data_psd(freqs)) * df
    print("Template norm (h|h):", sigmasq.real)

    return template_match



# calculate matched filter between actual template
def calculate_matched_filter(template, total_data, det, t_amount=4):
    #change first vairable from template_p (time domain) to FD_waveform (template) for consistency 
    """Calculates the best-fit template phase, offset, d_eff, horizon, and SNRmax
    values for both detectors on a given stretch of data given the desires
    template. Also can plot template shifts/residual data and print the
    parameters found.

    Args:
        template_p (ndarray): plus polarization of template
        event (dict): subdictionary of BBH-events containing event parameters
        t_amount (float): amount of time (s) around event to calcualate the
            matched filter
        total_data (dict): dict containing original and whitenbp strain data
        make_plots (bool, optional): if True, plot template shifts,
            whitened data, and residuals for each det.
        print_vals (bool, optional): if True, output params found

    Returns:
        dict: dictionary of parameters found and residual data for each detector
    """
    #template_p= np.fft.irfft(template)


    # these specific values are defined in the paper
    fband = [35.0, 350.0]

    time = total_data['time']
    time_center = total_data['time_center']
    dt = total_data['dt']
    fs = total_data['fs']
    large_data_psds = total_data['large_data_psds']


    # these dictionaries will be returned with our matched filter data and
    # residuals
    filter_data = {'H1': {}, 'L1': {}}

    # amount of data we want to calculate matched filter SNR over- up to 32s
    data_time_window = time[len(time) - 1] - time[0] - (32 - t_amount)

    time_filter_window = np.where((time <= time_center + data_time_window * .5) &
                                  (time >= time_center - data_time_window * .5))
    time_filtered = time[time_filter_window]
    #template_p = template_p[time_filter_window]

    # define the template using only the plus polarization
    #this is in time domain
   # template = template_p

    # loop over the detectors'
    strain = total_data[det]['strain'][time_filter_window]
    strain_whiten = total_data[det]['strain_whiten'][time_filter_window]
    strain_whitenbp = total_data[det]['strain_whitenbp'][time_filter_window]
    data_psd = large_data_psds[det]

    # save the time for later
    filter_data[det]['time'] = time_filtered

    # find the best fit phase, offset, d_eff, horizon
    SNRmax, timemax, d_eff, horizon, phase, offset = matched_filter(
        template, strain, time_filtered, data_psd, fs)

    # save these vals for later
    filter_data[det]['SNR'] = SNRmax
    filter_data[det]['d_eff'] = d_eff
    filter_data[det]['phase'] = phase
    filter_data[det]['offset'] = offset

    # get residuals and whitened data/template
    template_wbp = get_shifted_data(
        template, fband, filter_data[det], data_psd, dt)
    
    
    
    return template_wbp, strain_whitenbp, time_filtered - time_center, SNRmax, 1 / d_eff, phase


# wrapper function for matched filter
def wrapped_matched_filter(params, GW_signal, det):
    return opt_template(get_template(params, GW_signal.dictionary), GW_signal.dictionary, det, GW_signal.t_min, GW_signal.t_max)

def residual_func(data, fit):
    return data-fit