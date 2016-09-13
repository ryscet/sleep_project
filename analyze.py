#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:48:01 2016

@author: ryszardcetnarski
"""

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd
import cross_correlate as cs

# Change this to script function and namecross correlation!!!!!!!!!!!----------!!!!!


def RunAll():
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
#    explore_eeg(channel)
        explore_eeg(channel)
    #print(channel)

#Parse all psg channels using
#def parallel_explore():
#    num_cores = multiprocessing.cpu_count()
#    Parallel(n_jobs=num_cores)(delayed(RunAll)(name) for name in ['O2-A1', 'O1-A2', 'F4-A1', 'C4-A1', 'C3-A2'])
#    


def explore_slices(channel):
    print(0)
    psg_slices = np.load('parsed_data/psg_'+ channel + '_slices.npy')
    neuroon_slices = np.load('parsed_data/neuroon_slices.npy')
    
    psg = pd.Series.from_csv('parsed_data/psg_' + channel + '.csv')
    neuroon = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
    print(1)
    # FFT for psg
    fig, axes = plt.subplots(2, figsize = (40,40))
    fig.suptitle(channel)
    #Calc fft for the whole signal
    f_psg, Pxx_psg = signal.welch(psg, 200, nperseg=1024, noverlap = 128)
    # Calc fft for the slices
    f_psg_slices, Pxx_psg_slices = signal.welch(psg_slices, 200, nperseg=256, noverlap = 128) 
    # Plot the power spectrum of the signal
    axes[0].plot(f_psg[0 : np.argmax(f_psg > 60)], np.log(Pxx_psg)[0 : np.argmax(f_psg > 60)], c = 'r')
    # Plot the power spectrum of the slices
    g = sns.tsplot(data=np.log(Pxx_psg_slices[:,0 : np.argmax(f_psg_slices > 60)]), time = f_psg_slices[0 : np.argmax(f_psg_slices > 60)], err_style="unit_traces", ax = axes[1], color = 'r')    

    # FFT for neuroon
    fig2, axes2 = plt.subplots(2, figsize = (40,40))
    fig2.suptitle('Neuroon')
    # Calc fft for the whole signal
    f_non, Pxx_non = signal.welch(neuroon, 125, nperseg=1024, noverlap = 128)
    # Calc fft for the slices
    f_no_slices, Pxx_no_slices = signal.welch(neuroon_slices, 125, nperseg=256, noverlap = 128)
    # Plot the power spectrum of the signal
    axes2[0].plot(f_non, np.log(Pxx_non), c = 'g')
    # Plot the power spectrum of the slices
    g = sns.tsplot(data=np.log(Pxx_no_slices[:,0 : np.argmax(f_no_slices > 60)]), time = f_no_slices[0 : np.argmax(f_no_slices > 60)], err_style="unit_traces",  condition = 'no', ax = axes2[1], color = 'g')    
    print(2)
    for ax1, ax2 in zip(axes, axes2):
        ax1.set_ylabel('Power density')
        ax2.set_ylabel('Power density')
        
        ax1.set_xlabel('Frequency')
        ax2.set_xlabel('Frequency')
        
    fig.savefig('figures/psg_fft_'+ channel +'.pdf')
    fig2.savefig('figures/neuroon_fft.pdf')
    
    
def explore_eeg(channel):
    #fig, axes = plt.subplots()
    
    #psg = pd.Series.from_csv('parsed_data/psg_' + channel + '.csv')
    #neuroon = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
    
    psg = pd.Series.from_csv('parsed_data/psg_' + channel + '.csv')
    neuroon = pd.Series.from_csv('/Users/ryszardcetnarski/Desktop/sleep_project/neuroon_signals/night_01/neuroon_accel.csv', header =0)
    
    
    fig, axes = plt.subplots()
    axes.plot(psg,'r', label = 'psg', alpha = 0.5)
    axes.plot(neuroon, 'b', label = 'neuroon', alpha = 0.5)
    plt.legend()
    
    #psg_slices = np.load('parsed_data/1_hour_psg_' + channel + '_slices.npy')
    #neuroon_slices = np.load('parsed_data/1_hour_neuroon_slices.npy')
    
    global psg
    global neuroon
    
    psg2 = psg.rolling(window=120,center=False).mean()
    neuroon2 = neuroon.rolling(window=120,center=False).mean()
    

    
    #psg2 = psg2 / psg2.mean()
    #neuroon2 = neuroon2 / neuroon2.mean()
    
    neuroon2 = neuroon2.resample('10ms').mean()
    psg2 = psg2.resample('10ms').mean()
    
    slice_len = 1000
    for i in range(1, len(neuroon / slice_len)):
        
        print(i)
        
        st = int(len(neuroon) * 0.43)
        
        neuroon3 = neuroon2.iloc[st + ((i-1) * slice_len):st + (i * slice_len)]
        psg3 = psg2.iloc[st + ((i-1) * slice_len) : st + (i * slice_len)]
                         
        psg3 = (psg3 - psg3.mean()) / psg3.std()
        neuroon3 = (neuroon3 - neuroon3.mean()) / neuroon3.std()
        
        
        
        z = cs.CrossCorrelate({'psg':psg3,'neuroon':neuroon3}, 600, 1, channel)
        
        if i > 10:
            break
    
    # Todo, repeat the cc on many short samples and save the averages
    
  #  axes.plot(neuroon2, color = 'b', alpha = 0.7)
 #   axes.plot(psg2, color = 'r', alpha = 0.7)
    
