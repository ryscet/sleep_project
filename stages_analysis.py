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
from psg_edf_2_hdf import hdf_to_series as load_channel
from psg_edf_2_hdf import hdf_to_spectrum_dict as load_spectrum

plt.style.use('ggplot')

 

def compare_neuroon_psg():

    neuroon_spectra, neuroon_frequency =  load_spectrum('neuroon')
    
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
        fig, axes = plt.subplots(4, figsize = (10,20), sharey = True)
        fig.suptitle(channel, fontweight = ('bold'))
        axes[3].set_xlabel('Frequency (Hz)')
        fig.text(0.04, 0.5, 'power spectral density', va='center', rotation='vertical')

        print(channel)
        # prepare slices of signal comprising all times a sleep stage was identified
        psg_spectra, psg_frequency =  load_spectrum(channel)
       
       # Plot the results
        plot_spectra_by_device(psg_spectra, psg_frequency, axes, 'psg', 'seagreen')
        
        plot_spectra_by_device(neuroon_spectra, neuroon_frequency, axes, 'neuroon', 'cornflowerblue')
       
        fig.savefig('figures/device_split/psg_' + channel+ '_neuroon_fft.pdf')
        raise_window()
        
def plot_spectra_by_device(spectra, frequency, axes, psg_or_noo,color):
    
    
    axes_dict = {'N1' : 0, 'N2':1, 'N3' : 2, 'rem' : 3}
    for stage_name, spectrum in spectra.items():
        if(stage_name != 'wake'):
            max_idx = np.argmax(frequency > 50)

            sns.tsplot(data=np.log(spectrum[:, 0: max_idx]), time = frequency[0 : max_idx], condition = psg_or_noo, legend = True, color=color,err_style="unit_traces", ax = axes[axes_dict[stage_name]])
            axes[axes_dict[stage_name]].set_ylabel(stage_name)
        

def compare_stages():

    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
        fig, axes = plt.subplots()
        fig.suptitle(channel, fontweight = ('bold'))
        axes.set_xlabel('Frequency (Hz)')
        fig.text(0.04, 0.5, 'power spectral density', va='center', rotation='vertical')


        
        print(channel)
        psg_spectra, psg_frequency =  load_spectrum(channel)


        plot_spectra_by_stage(psg_spectra, psg_frequency, axes, 'psg')

        fig.savefig('figures/stage_split/' +channel + '_by_phase.pdf')
        
    fig, axes = plt.subplots()
    fig.suptitle('neuroon', fontweight = ('bold'))
    axes.set_xlabel('Frequency (Hz)')
    fig.text(0.04, 0.5, 'power spectral density', va='center', rotation='vertical')    
    
    neuroon_spectra, neuroon_frequency = load_spectrum('neuroon')
    plot_spectra_by_stage(neuroon_spectra, neuroon_frequency, axes, 'neuroon')
 
    fig.savefig('figures/stage_split/neuroon_by_phase.pdf')
    
def compare_electrodes():
    fig, axes = plt.subplots(4, figsize = (10,20), sharey = True)
    fig.suptitle('psg electrodes comparison', fontweight = ('bold'))
    
    axes[3].set_xlabel('Frequency (Hz)')
    
    
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
  
        print(channel)
        # Load the signal from a channel recorded by psg
        psg_spectra, psg_frequency =  load_spectrum(channel)

        # Plot the results
        plot_spectra_by_electrode(psg_spectra, psg_frequency, axes, channel)

        
        
    fig.savefig('figures/electrode_split/psg_electrode_diffs')


        
def plot_spectra_by_stage( spectra, frequency, axes, psg_or_noo):
    
    max_idx = np.argmax(frequency  >50)

    for stage_name, spectrum in spectra.items():
        if(stage_name != 'wake'):

            #sns.tsplot(data=np.log(spectrum[:, 0: max_idx]), time = frequency[0, 0 : max_idx], condition = stage_name , legend = True, color=color_dict , ax = axes)        
        #    for pxx, f in zip(spectrum, frequency):
              #  axes.plot(f, np.log(pxx), color = color_dict[stage_name], alpha = 0.1)
            grand_avg = np.log(spectrum).mean(axis = 0)
            std =np.log(spectrum).std(axis = 0)
            
            # Select only part of frequencies
            grand_avg = grand_avg[0:max_idx]
            std = std[0:max_idx]
            frequency = frequency[0:max_idx]
            
            axes.plot( frequency, grand_avg ,color = stage_color_dict[stage_name], label = stage_name)
            
            axes.fill_between(frequency, grand_avg - std, grand_avg + std, color = stage_color_dict[stage_name], alpha = 0.2)

    plt.legend()
    
def plot_spectra_by_electrode(spectra, frequency, axes, channel):
    
    max_idx = np.argmax(frequency >50)

    for stage_name, spectrum in spectra.items():
        if(stage_name != 'wake'):

            #sns.tsplot(data=np.log(spectrum[:, 0: max_idx]), time = frequency[0, 0 : max_idx], condition = stage_name , legend = True, color=color_dict , ax = axes)        
        #    for pxx, f in zip(spectrum, frequency):
              #  axes.plot(f, np.log(pxx), color = color_dict[stage_name], alpha = 0.1)
            grand_avg = np.log(spectrum).mean(axis = 0)
            std =np.log(spectrum).std(axis = 0)
            
            # Select only part of frequencies
            grand_avg = grand_avg[0:max_idx]
            std = std[0:max_idx]
            frequency = frequency[0:max_idx]
            
            axes[axes_dict[stage_name]].plot( frequency, grand_avg ,color = electrode_color_dict[channel], label = channel)
            
            axes[axes_dict[stage_name]].fill_between(frequency, grand_avg - std, grand_avg + std, color = electrode_color_dict[channel], alpha = 0.2)
            
            axes[axes_dict[stage_name]].set_ylabel(stage_name + ' power density')
            
        plt.legend()
        
        
if __name__ == '__main__':  
    try:
        neuroon_hipno
        print('loaded')
    except NameError:
        print('not found, loading')
        psg_hipno = ph.prep_for_phases(ph.parse_psg_stages())
        neuroon_hipno = ph.prep_for_phases(ph.parse_neuroon_stages())
        stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'y' }
        electrode_color_dict = {'O2-A1' : 'purple', 'O1-A2' :'mediumorchid', 'F4-A1' : 'royalblue', 'F3-A2' : 'dodgerblue', 'C4-A1' : 'seagreen', 'C3-A2' : 'darkseagreen' }
        axes_dict = {'N1' : 0, 'N2':1, 'N3' : 2, 'rem' : 3}
    
    compare_neuroon_psg()
    compare_stages()
    compare_electrodes()
    
#def 

    #    err_style="unit_traces"
        #time = frequency[:, 0:max_idx],