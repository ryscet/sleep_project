#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:53:59 2016

@author: user
"""

import numpy as np
import pandas as pd

import parse_hipnogram as ph
import intersect_hipno as ih

num_to_stage = {5: 'wake', 1 : 'rem', 2 :'N1', 3 : 'N2', 4: 'N3'}
# Decide whther to do the permutation on the 30 sec slices or on the already binned data of continous events

# iterate over rows of neuroon hipnogram and randomly assign to them a stage with replacement from a list made of stages that were used (time empirical-bootstrap?)
# Try the same from a random uniform population, 30% for each stage and from random population with theoretical paramete4rs based on literature - i.e. huw much of each sleep stage usually

def permute_neuroon():
    num_perm = 1000
    psg_hipno = ph.parse_psg_stages()
    noo_hipno = ph.parse_neuroon_stages()
    
    # Get the start and end of the time window covered by both hipnograms
    start = noo_hipno.index.searchsorted(psg_hipno.index.get_values()[0])
    end = psg_hipno.index.searchsorted(noo_hipno.index.get_values()[-1])
    
    # Trim hipnograms to the common time window so the confusion matrix calculations are accurate
    # +1 and -1 because events got cut in half, resulting in ends without starts
    noo_hipno = noo_hipno.ix[start +1::]
    psg_hipno = psg_hipno.ix[0:end -1]
    
    for i in range(num_perm):
        # Make a copy just in case
        permuted_neuroon = noo_hipno.copy()
        #Permute the sequency of stages (without replacement)
        
        permuted_neuroon.loc[:, 'stage_num'] = shuffle_in_unison_inplace(noo_hipno['stage_num'].as_matrix())
        
        permuted_neuroon.loc[:, 'stage_name'] =  permuted_neuroon.loc[:, 'stage_name'].replace(num_to_stage)
        
        new_list = [y[i:i+2] for i in range(0, len(y), 2)]
        
        
        shuffle_in_unison_inplace(noo_hipno['stage_num'].as_matrix())

    psg_total = calc_times(ph.prep_for_phases(psg_hipno.copy()))
    neuroon_total= calc_times(ph.prep_for_phases(noo_hipno.copy()))
    numpy.random.permutation(x)
    
def shuffle_in_unison_inplace(list_of_columns):
    
    #Bin the list into two row pairs and permute these pairs. This way, the start and end stage will always be the same (if there were to be a stage that starts as N2 and ends as rem, counting their time wouldn't work)
    for idx, column in enumerate(list_of_columns):
        list_of_columns[idx] = np.array([np.array(column[i:i+2]) for i in range(0, len(column), 2)])
    return list_of_columns 
    
    assert len(list_of_columns[0]) == len(list_of_columns[1])
    # Create prmutation to shuffle two lists
    p = numpy.random.permutation(len(list_of_columns[0]))
    
    return list_of_columns[0][p], list_of_columns[1][p]