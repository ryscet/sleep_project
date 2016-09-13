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
import nupy as np

plt.style.use('ggplot')

if __name__ == '__main__':  

    try:
        neuroon_hipno
        print('loaded')
    except NameError:
        print('not found, loading')
        psg_hipno = ph.prep_for_phases(ph.parse_psg_stages())
        neuroon_hipno = ph.prep_for_phases(ph.parse_neuroon_stages())
        neuroon_signal = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
        stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'y' }
        electrode_color_dict = {'O2-A1' : 'purple', 'O1-A2' :'mediumorchid', 'F4-A1' : 'royalblue', 'F3-A2' : 'dodgerblue', 'C4-A1' : 'seagreen', 'C3-A2' : 'darkseagreen' }


def parse_stages():
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
        psg_signal =  pd.Series.from_csv('parsed_data/psg_' + channel + '.csv')
        psg_slices, psg_spectra, psg_frequency =  sa.make_stage_slices(200, psg_hipno, psg_signal, channel)
        
        dict_to_hdf(_dict)
            
        
        
        break


def dict_to_hdf(_dict, path):
    path = 'parsed_data/hdf/psg_database.h5'
 
    with h5py.File(path, 'w') as hf:

        for key, value in _dict.items():
            print(key)
            hf.create_dataset(key, data = value)




##Pca decomposition into first two components
#sklearn_pca = sklearnPCA(n_components=2)
#pcs = sklearn_pca.fit_transform(sigs_array)
#
#
#ax2.scatter(pcs[:,0], pcs[:,1], c = mask_array, cmap = 'jet', s = 60, marker = 'o')
