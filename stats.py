#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:28:03 2022

@author: emilia
"""
import numpy as np
from scipy.stats import shapiro, ttest_ind, ttest_1samp, mannwhitneyu, wilcoxon


def run_contrast_single_sample(data,h_mean,alt='greater'):
    # check if normal distribution
    s,p = shapiro(data)
    if p >= .05:
        # parametric statistics
        stat,p_val = ttest_1samp(data,h_mean,alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' %(stat,p_val))
    else:
        # try a log transform if data positive
        s,p = shapiro(np.log(data))
        non_neg = np.all(data>0)
        if np.logical_and(p >= .05,non_neg):
            stat,p_val = ttest_1samp(np.log(data),np.log(h_mean),alternative=alt)
            print('......One-sample t-test,log-transformed data: '\
                  +'stat= %.2f, p-val = %.6f' %(stat,p_val))
        else:
            # do non-parametric statistics
            stat,p_val =  wilcoxon(data-h_mean,alternative=alt)
            print('......Wicoxon test: stat= %.2f, p-val = %.6f' %(stat,p_val))


def run_contrast_paired_samples(data1,data2,alt='greater'):
    # check if normal distribution
    s,p = shapiro(data1-data2)      
    if p>=.05:
        # parametric stats
        stat,p_val = ttest_1samp(data1-data2,0,alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' %(stat,p_val))
    else:
        # try log-transform
        non_neg = np.all(data1-data2>0)
        s,p = shapiro(np.log(data1-data2))
        if np.logical_and(p >= .05,non_neg):
            stat,p_val = ttest_1samp(np.log(data1-data2),0,alternative=alt)
            print('......One-sample t-test,log-transformed data: '\
                  +'stat= %.2f, p-val = %.6f' %(stat,p_val))
        else:
            # do non-parametric statistics
            stat,p_val =  wilcoxon(data1-data2,alternative=alt)
            print('......Wicoxon test: stat= %.2f, p-val = %.6f' %(stat,p_val))        
              

def run_contrast_unpaired_samples(data1,data2,alt='greater'):
    # check if normal distribution
    s1,p1 = shapiro(data1)     
    s2,p2 = shapiro(data2)      

    if np.logical_and(p1>=.05,p2>=.05):
        # parametric stats
        stat,p_val = ttest_ind(data1,data2,alternative=alt)
        print('......Indiv. samples t-test: stat= %.2f, p-val = %.6f' %(stat,p_val))
    else:
        # try log-transform
        non_neg = np.logical_and(np.all(data1>0),np.all(data1>0))
        s1,p1 = shapiro(np.log(data1))     
        s2,p2 = shapiro(np.log(data2))   
        if np.all((p1>= .05,p2>=.05,non_neg)):
            stat,p_val = ttest_ind(np.log(data1),np.log(data2),alternative=alt)
            print('......Indiv. samples t-test,log-transformed data: '\
                  +'stat= %.2f, p-val = %.6f' %(stat,p_val))
        else:
            # do non-parametric statistics
            stat,p_val =  mannwhitneyu(data1,data2,alternative=alt)
            print('......Wicoxon test: stat= %.2f, p-val = %.6f' %(stat,p_val))   