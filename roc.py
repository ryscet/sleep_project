#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:17:01 2016

@author: user
"""
import parse_hipnogram as ph
import intersect_hipno as ih
from collections import OrderedDict 
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib as mpl
   #print(tabulate([[F, p, between_group_sumsq, df_between,  within_group_sumsq, df_within]],
            #         ['F-value','p-value','effect sss','effect df','error sss', 'error df'], tablefmt="grid"))


# Make sure common times and psg timeas have the same res (they dont)
stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'y' }

def run_permute():
    fig, axes = plt.subplots()
    
    orig_confusion_matrixes, orig_cl_params = get_roc(permute = False)

    for i in range(30):
        print(i)
        all_confusion_matrixes, all_cl_params = get_roc(permute = True)
        tp, fp = all_confusion_matrixes['N2']['true_positive'], all_confusion_matrixes['N2']['false_positive']
        axes.plot(fp, tp, color = 'r', marker = 'o', alpha = 0.5)
    
    axes.plot(orig_confusion_matrixes['N2']['false_positive'], orig_confusion_matrixes['N2']['true_positive'], color = 'b', marker = 'o', alpha = 1)
    
    axes.plot([0,1], [0,1], color = 'black', linestyle = '--', alpha = 0.5)
    axes.set_xlabel('false_positive')
    axes.set_ylabel('true_positive')
    
def get_roc(permute = False):
    psg_hipno = ph.parse_psg_stages()
    noo_hipno = ph.parse_neuroon_stages(permute)
    
    # Get the start and end of the time window covered by both hipnograms
    start = noo_hipno.index.searchsorted(psg_hipno.index.get_values()[0])
    end = psg_hipno.index.searchsorted(noo_hipno.index.get_values()[-1])
    
    # Trim hipnograms to the common time window so the confusion matrix calculations are accurate
    # +1 and -1 because events got cut in half, resulting in ends without starts
    noo_hipno = noo_hipno.ix[start +1::]
    psg_hipno = psg_hipno.ix[0:end -1]
    
    psg_total = calc_times(ph.prep_for_phases(psg_hipno.copy()))
    neuroon_total= calc_times(ph.prep_for_phases(noo_hipno.copy()))
    
    neuroon_correct = ih.get_hipnogram_intersection(psg_hipno.copy(), noo_hipno.copy(), 0)

    all_confusion_matrixes = {}
    all_cl_params = {}
    for stage in list(neuroon_total.keys()):
        
         #The time neuroon said it was not this stage
        neuroon_negative = np.array([neuroon_total[key] for key in list(neuroon_total.keys()) if key != stage]).sum()
        # The time psg said it was not this stage
        subset = list(neuroon_total.keys())
        subset.extend(['N1', 'wake'])# These keys are not in neuroon

        psg_negative = np.array([psg_total[key] for key in subset if key != stage]).sum()
        
        true_positive = neuroon_correct[stage] / psg_total[stage]
        true_negative = neuroon_negative/ psg_negative

        false_positive = (neuroon_total[stage] - neuroon_correct[stage]) / psg_negative 
        false_negative =  (psg_total[stage] - neuroon_correct[stage]) / neuroon_negative 
#        
        confusion_matrix ={'true_positive' : true_positive, 'false_positive' : false_positive,'false_negative':false_negative, 'true_negative' : true_negative}

        all_confusion_matrixes[stage] = confusion_matrix
        
        cl_params = {}
        cl_params['precision'] = true_positive / (true_positive + false_positive)
        cl_params['recall'] = true_positive / (true_positive + false_negative)
        cl_params['accuracy'] = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        
        all_cl_params[stage] = cl_params
        
    
   # plot_roc(all_confusion_matrixes, all_cl_params)
    
    return all_confusion_matrixes, all_cl_params

        
def plot_roc(all_confusion_matrix, all_cl_params):
    
        roc_fig, roc_axes = plt.subplots()
        
        for stage in list(all_confusion_matrix.keys()):
           
            confusion_matrix= all_confusion_matrix[stage]
            cl_params = all_cl_params[stage]
            
            fig, ax= plt.subplots() 
            fig.suptitle('classification performance for %s'%stage, fontweight = 'bold')
            #plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
            col_labels=['psg positive','psg negative']
            row_labels=['neuroon\n positive','neuroon\n negative']
            table_vals=[['TP: %.2f'%confusion_matrix['true_positive'],'FP: %.2f'%confusion_matrix['false_positive']],['FN: %.2f'%confusion_matrix['false_negative'], 'TN: %.2f'%confusion_matrix['true_negative']]]
            # the rectangle is where I want to place the table
            the_table = ax.table(cellText=table_vals,
                              colWidths = [0.2]*2,
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center')
            ax.text(0.75,0.60,'accuracy: %.2f'%cl_params['accuracy'] ,size=14)
            ax.text(0.75, 0.50,'precision: %.2f'%cl_params['precision'],size=14)
            ax.text(0.75,0.40,'recall: %.2f'%cl_params['recall'],size=14)
            the_table.scale(1, 2.5)
            ax.xaxis.set_visible(False) 
            ax.yaxis.set_visible(False)
            fig.savefig('figures/roc/cm_%s.pdf'%stage)


            roc_axes.plot(confusion_matrix['false_positive'], confusion_matrix['true_positive'], marker= 'o', color = stage_color_dict[stage], label = stage)
        
        roc_axes.plot([0,1], [0,1], color = 'black', linestyle = '--', alpha = 0.5)
        roc_axes.set_xlabel('false_positive')
        roc_axes.set_ylabel('true_positive')
        roc_axes.legend()
        
        roc_fig.savefig('figures/roc/roc.pdf')
        
        raise_window()
    

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
            
            
            
