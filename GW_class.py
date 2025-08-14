import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as integrate
from scipy.interpolate import interp1d
from pycbc.conversions import mchirp_from_mass1_mass2, spin1z_from_mass1_mass2_chi_eff_chi_a, spin2z_from_mass1_mass2_chi_eff_chi_a
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from constants import *
from template import waveform
from matched_filter import *
import pickle
from signal_processing import *


class GWSignals:

    def __init__(self, ref_params, dictionary):
        # dictionary which contains data for event
        self.dictionary = dictionary

        # reference parameters
        #add amplitude into reference params 
        self.mass1, self.mass2, self.chiPlus, self.chiMinus = ref_params
        self.chirp = mchirp_from_mass1_mass2(self.mass1, self.mass2)
        self.ratio = self.mass2 / self.mass1
        self.chi1 = spin1z_from_mass1_mass2_chi_eff_chi_a(self.mass1, self.mass2, self.chiPlus, self.chiMinus)
        self.chi2 = spin2z_from_mass1_mass2_chi_eff_chi_a(self.mass1, self.mass2, self.chiPlus, self.chiMinus)  
        self.comp_params = np.array([self.mass1, self.mass2, self.chi1, self.chi2])

        # minimum / maximum mass parameters for sliders
        #add amp range here
        self.min_mass1 = self.mass1 - 5.0
        self.min_mass2 = self.mass2 - 5.0
        self.max_mass1 = self.mass1 + 5.0
        self.max_mass2 = self.mass2 + 5.0
        self.min_chirp = mchirp_from_mass1_mass2(self.min_mass1, self.min_mass2)
        self.max_chirp = mchirp_from_mass1_mass2(self.max_mass1, self.max_mass2)




# load data 
with open('data/GW150914_data_dict.pkl', 'rb') as f:
    GW150914_data = pickle.load(f)

with open('data/GW190521_data_dict.pkl', 'rb') as f:
    GW190521_data = pickle.load(f)

with open('data/GW200129_data_dict.pkl', 'rb') as f:
    GW200129_data = pickle.load(f)

with open('data/GW200224_data_dict.pkl', 'rb') as f:
    GW200224_data = pickle.load(f)

with open('data/GW200311_data_dict.pkl', 'rb') as f:
    GW200311_data = pickle.load(f)

with open('data/GW191109_data_dict.pkl', 'rb') as f:
    GW191109_data= pickle.load(f)

with open('data/GW190828_data_dict.pkl', 'rb') as f:
    GW190828_data= pickle.load(f)

with open('data/GW190519_data_dict.pkl', 'rb') as f:
    GW190519_data= pickle.load(f)

with open('data/simulated_GW.pkl', 'rb') as f:
    simulated_data = pickle.load(f)

# class instantiation for real GW events
GW150914 = GWSignals(signal_ref_params['GW150914'][1], GW150914_data)
GW190521 = GWSignals(signal_ref_params['GW190521'][1], GW190521_data)
GW200129 = GWSignals(signal_ref_params['GW200129'][1], GW200129_data)
GW200224 = GWSignals(signal_ref_params['GW200224'][1], GW200224_data)
GW200311 = GWSignals(signal_ref_params['GW200311'][1], GW200311_data)
GW191109 = GWSignals(signal_ref_params['GW191109'][1], GW191109_data)
GW190828= GWSignals(signal_ref_params['GW190828'][1], GW190828_data)
GW190519= GWSignals(signal_ref_params['GW190519'][1], GW190519_data)

# GW_simulated = GWSignals(signal_ref_params['GW150914'][1], GW150914_data)

GW_simulated = GWSignals(np.array([30., 20., 0., 0.]), simulated_data)



