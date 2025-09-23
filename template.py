'''Get gravitational wave time-domain template given parameters.'''


import numpy as np
from IMRPhenomD.IMRPhenomD import AmpPhaseFDWaveform, IMRPhenomDGenerateh22FDAmpPhase
import IMRPhenomD.IMRPhenomD_const as imrc
from scipy.interpolate import interp1d
import constants as c
from scipy.signal import resample
from scipy.signal.windows import tukey


# gravitational waveform class for simulated waveforms
class Waveform:
    
    def __init__(self, freqs, freqs_for_waveform):
        
       # frequency bins to generate frequency-domain waveform
        self.freqs_for_waveform = freqs_for_waveform

       #frequencies for FFt's
        self.freqs = freqs 
        
        # initialize and store frequency-like objects
        self.num_freqs_waveform = len(self.freqs_for_waveform)
        self.num_freqs = len(self.freqs)
        self.df = self.freqs[1] - self.freqs[0]
        self.freq_min = self.freqs[0]
        self.freq_max = self.freqs[-1]
        self.sampling_freq = 2 * self.freq_max

        # hyperbolic tangent window for iFFT, centered near f_min
        self.tanh_window = np.tanh(self.freqs_for_waveform - np.array([self.freq_min] * self.num_freqs_waveform))

        # time axis
        self.waveform_TD_full_shape = (self.num_freqs - 1) * 2
        self.times_full = np.arange(0, self.waveform_TD_full_shape/self.sampling_freq, 1/self.sampling_freq) + c.window_min
        self.times = self.times_full[(self.times_full >= c.window_min) & (self.times_full <= c.window_max)]
        self.Nt = self.times.shape[0]
        self.dt = self.times[1] - self.times[0]
        self.merger_index = np.argmin(np.abs(self.times))

    
    # reference frequency for waveform generation
    MfRef_in = 0.  # ref. freq. at peak amplitude in freq-domain
    
    # distance to source in meters
    distance = c.DL_SI
    
    # get (frequency-domain) h22 object for given parameters
    # h22 includes amplitude and phase as array
    def get_h22(self, params, phic):
                
        # mass in solar masses
        m1, m2, chi1, chi2 = params
        
        # masses in kg
        m1_SI =  m1 * imrc.MSUN_SI
        m2_SI =  m2 * imrc.MSUN_SI

        # initialize amplitudes and times
        amp_imr = np.zeros(self.num_freqs_waveform)
        phase_imr = np.zeros(self.num_freqs_waveform)
        time_imr = np.zeros(self.num_freqs_waveform)
        timep_imr = np.zeros(self.num_freqs_waveform)

        #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
        h22 = AmpPhaseFDWaveform(self.num_freqs_waveform,self.freqs_for_waveform,amp_imr,phase_imr,time_imr,timep_imr,0.,0.)
        h22 = IMRPhenomDGenerateh22FDAmpPhase(h22,self.freqs_for_waveform,phic,Waveform.MfRef_in,m1_SI,m2_SI,chi1,chi2,Waveform.distance)

        return h22
    
    
    # get frequency-domain signal
    def get_FD_waveform(self, params, phic):
        h22 = self.get_h22(params, phic)
        waveform_FD = h22.amp * np.exp(-1.j * h22.phase)
        

        # apply hyperbolic tangent window
        windowed_waveform_FD = self.tanh_window * waveform_FD
        # pad with zeros down to DC component

        full_FD_waveform = np.zeros(self.num_freqs, dtype='complex')
        full_FD_waveform[-self.num_freqs_waveform:] = windowed_waveform_FD
        #full_FD_waveform[-self.num_freqs_waveform:] = waveform_FD
        return full_FD_waveform


    # inverse FFT waveform to go into time-domain
    def iFFT_waveform(self, waveform_FD):
        waveform_TD = np.fft.irfft(waveform_FD) 
        # set merger to t = 0
        #waveform_TD = np.roll(waveform_TD, self.merger_index - np.argmax(waveform_TD))[:self.Nt]
        waveform_TD = np.roll(waveform_TD, self.merger_index - np.argmax(waveform_TD))
        return waveform_TD
    

    # # get signal in time-domain given parameters
    # def get_TD_waveform(self, params, phic):
    #     waveform_FD = self.get_FD_waveform(params, phic)
    #     return self.iFFT_waveform(waveform_FD)


# instantiate waveform class for frequency bins (defined in constants.py)
waveform = Waveform(c.freqs, c.freqs_for_waveform)


def get_template(comp_params, data_dict):
    dt = data_dict['dt']
    fs = data_dict['fs']

    fig_template_tapered = waveform.get_FD_waveform(comp_params, 0.) 
    # fig_template = np.array([waveform.times, waveform.get_TD_waveform(comp_params, 0.0)]).T 
    # times_interp = np.linspace(waveform.times[0], waveform.times[-1], 6000)
    # waveform_interp = interp1d(fig_template[:, 0], fig_template[:, 1])(times_interp)
    # fig_template = np.array([times_interp, waveform_interp]).T[-3440:]
    # fig_template = fig_template[:, 1]

    # Downsample this data to 4096 Hz
    # fig_template = resample(fig_template, int(len(fig_template)/4) )

    # apply a Tukey window to taper the ends of the template
    ##taper_window = tukey(len(fig_template), alpha=.25)
    ##fig_template_tapered = fig_template* taper_window

    # -- Plot template before and after tapering
    ##template_time = np.arange(0.25, 0.25+len(fig_template_tapered)*dt,dt)

    # # Now we need to pad this with 0s to make it the same amount of time as the data
    # halfdatalen = int(16*fs)
    # begin_add = halfdatalen - len(fig_template_tapered)

    # # add last 2048 seconds
    # fig_template_tapered = np.append(fig_template_tapered, (halfdatalen * [0]))
    # # add beginning- almost 2048 seconds
    # fig_template_tapered = np.append((int(begin_add) * [0]), fig_template_tapered)

    return fig_template_tapered





