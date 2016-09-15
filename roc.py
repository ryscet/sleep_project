#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:17:01 2016

@author: user
"""
import parse_hipnogram as ph
import intersect_hipno as ih
from collections import OrderedDict 


# Make sure common times and psg timeas have the same res (they dont)

def get_roc():
    psg_hipno = ph.parse_psg_stages()
    noo_hipno = ph.parse_neuroon_stages()
    
    psg_times = calc_times(ph.prep_for_phases(psg_hipno.copy()))
    neuroon_times = calc_times(ph.prep_for_phases(noo_hipno.copy()))
    
    a, common_times = ih.get_hipnogram_intersection(psg_hipno.copy(), noo_hipno.copy(), 0)
    

def calc_times(hipno):
    
    stage_durations = OrderedDict()

    for name, sleep_stage in hipno.groupby('stage_name'):
    
        for idx, stage_event in sleep_stage.iterrows():
            duration = (stage_event['ends'] -  stage_event['starts']).total_seconds()
            
            if stage_event['stage_name'] not in stage_durations.keys():
                # Divide to get minutes
                stage_durations[stage_event['stage_name']] = duration
            else:
                stage_durations[stage_event['stage_name']] += duration
    
    
    return stage_durations
            
            
            
