#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 23:27:44 2016

@author: ryszardcetnarski
"""

import parse_hipnogram as ph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sig
import seaborn as sns

#hipno = ph.prep_for_phases(ph.parse_psg_stages())
#signal = pd.Series.from_csv('parsed_data/psg_' + channel + '.csv')

hipno = ph.prep_for_phases(ph.parse_neuroon_stages())
signal = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')

def make_stage_slices():
    channel = 'C3-A2'
    sampling_rate = 125 #!!!!!!-----!!!!----!!!!

    #noo_hipno = ph.prep_for_phases(ph.parse_neuroon_stages())
    
    slices = {}
    spectra = {}
    frequency = {}
    for name, sleep_stage in hipno.groupby('stage_name'):
        stage_spectra = []
        stage_slices = []
        stage_freqs = []

        for idx, phase_event in sleep_stage.iterrows():
            _slice = np.array(signal.loc[phase_event['starts'] : phase_event['ends']])
            if(_slice.size !=  0):
                stage_slices.append(_slice)
                   
                freqs, pxx = sig.welch(_slice, sampling_rate, nperseg=256, noverlap = 128) 
                stage_spectra.append(pxx)
                stage_freqs.append(freqs)
            
        slices[name] = stage_slices
        spectra[name] = np.array(stage_spectra)
        frequency[name] = np.array(stage_freqs)
        
    return slices, spectra, frequency
    
def plot_spectra_by_phase():
    fig = plt.figure()
    slices, spectra, freqs = make_stage_slices()
    
    color_dict = {'N1' : 'b', 'N2' :'r', 'N3' : 'g', 'rem' : 'y', 'wake' : 'm' }
    for (stage_name, spectrum), (stage_name2, frequency) in zip(spectra.items(), freqs.items()):
        if(stage_name != 'wake'):
            max_idx = np.argmax(frequency[0,:] >50)

            sns.tsplot(data=np.log(spectrum[:, 0: max_idx]), time = frequency[0, 0 : max_idx], condition = stage_name, legend = True, color=color_dict, err_style="unit_traces")
        
        
        #time = frequency[:, 0:max_idx],