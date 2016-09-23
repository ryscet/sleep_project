# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:50:21 2016

@author: user
"""

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

import stages_analysis as sa
import parse_hipnogram as ph

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from pylab import Rectangle

from psg_edf_2_hdf import hdf_to_spectrum_dict as load_spectrum

plt.style.use('ggplot')


def run_pca():
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
        pca_stages(channel)

def pca_stages(channel):
    fig, axes = plt.subplots()
    fig.suptitle(channel, fontweight = 'bold')
    spectra, frequency =  load_spectrum(channel)
    
    flat = np.concatenate([spectra['N1'],spectra['N2'],spectra['N3'], spectra['rem']], axis = 0)
                            
    n1 = ['royalblue' for i in spectra['N1'][:,0]]
    n2 = ['forestgreen' for i in spectra['N2'][:,0]]
    n3 = ['coral' for i in spectra['N3'][:,0]]
    rem = ['plum' for i in spectra['rem'][:,0]]
     
    color = list(itertools.chain.from_iterable([n1,n2,n3,rem]))

    sklearn_pca = sklearnPCA(n_components=2)
    pcs = sklearn_pca.fit_transform(flat)    
    

    y = axes.scatter(pcs[:,0],pcs[:,1] , c = color, alpha = 0.7, s = 40, edgecolors = 'w')
    
    axes.annotate(sklearn_pca.explained_variance_ratio_,xy= (1.0,1.0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

    raise_window()
    
    axes.set_xlabel('1st component')
    axes.set_ylabel('2nd component')
#    plt.legend()    
    
        # make the legend
    p1 = Rectangle((0, 0), 1, 1, fc="royalblue")
    p2 = Rectangle((0, 0), 1, 1, fc="forestgreen")
    p3 = Rectangle((0, 0), 1, 1, fc="coral")
    p4 = Rectangle((0, 0), 1, 1, fc="plum")
    plt.legend((p1, p2, p3, p4), ('N1','N2','N3', 'rem'))
    
    fig.savefig('figures/pca/'+channel+'_pca.pdf')
    
def pca_stages_neuroon():
    fig, axes = plt.subplots()
    fig.suptitle('neuroon', fontweight = 'bold')
    spectra, frequency =  load_spectrum('neuroon')
    
    flat = np.concatenate([spectra['N2'],spectra['N3'], spectra['rem']], axis = 0)
                            
    n2 = ['forestgreen' for i in spectra['N2'][:,0]]
    n3 = ['coral' for i in spectra['N3'][:,0]]
    rem = ['plum' for i in spectra['rem'][:,0]]
     
    color = list(itertools.chain.from_iterable([n2,n3,rem]))

    sklearn_pca = sklearnPCA(n_components=2)
    pcs = sklearn_pca.fit_transform(flat)    
    

    y = axes.scatter(pcs[:,0],pcs[:,1] , c = color, alpha = 0.7, s = 40, edgecolors = 'w')
    
    axes.annotate(sklearn_pca.explained_variance_ratio_,xy= (1.0,1.0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

    raise_window()
    
    axes.set_xlabel('1st component')
    axes.set_ylabel('2nd component')
#    plt.legend()    
    
        # make the legend
    p2 = Rectangle((0, 0), 1, 1, fc="forestgreen")
    p3 = Rectangle((0, 0), 1, 1, fc="coral")
    p4 = Rectangle((0, 0), 1, 1, fc="plum")
    plt.legend((p2, p3, p4), ('N2','N3', 'rem'))
    
    fig.savefig('figures/pca/neuroon_pca.pdf')
    


#[random.uniform(0.8, 1.2)for i in pcs[:,0]]

##Pca decomposition into first two components
#sklearn_pca = sklearnPCA(n_components=2)
#pcs = sklearn_pca.fit_transform(sigs_array)
#
#
#
