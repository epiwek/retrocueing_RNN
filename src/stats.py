#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:28:03 2022

@author: emilia
"""
import numpy as np
from scipy.stats import shapiro, ttest_ind, ttest_1samp, mannwhitneyu, wilcoxon
import pycircstat

def cohens_d(data1,data2):
    # calculate the size of the datasets
    n1, n2 = len(data1), len(data2)
    # calculate the variance of the samples
    s1 = np.var(data1, ddof=1) 
    s2 = 1 if n2==1 else np.var(data2, ddof=1) # set to 1 if data2 has only 1 point
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(data1), np.mean(data2)
    # calculate the effect size
    return (u1 - u2) / s


def cohens_d_circular(data1):
    # calculate the  standard deviation
    sd = pycircstat.descriptive.std(data1)
    # calculate the mean
    u = pycircstat.descriptive.mean(data1)
    
    # calculate the effect size
    return u / sd


from scipy.stats import rankdata
def matched_rank_biserial_corr(data1,data2):
    # calculate the difference
    diff = data1-data2
    #rank the difference
    ranks = rankdata(diff)
    # get the sum of the negative ranks
    W_neg = (ranks[np.where(diff<0)[0]]).sum()
    W_pos = (ranks[np.where(diff>=0)[0]]).sum()
    # get the total sum of ranks
    W_total = W_neg+W_pos
    
    # the direction willbe prop_favourable to the hypotehesis - unfavourable
    return W_neg/W_total - W_pos/W_total


def run_contrast_single_sample(data,h_mean,alt='greater'):
    # check if normal distribution
    s,p = shapiro(data)
    if p >= .05:
        # parametric statistics
        stat,p_val = ttest_1samp(data,h_mean,alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' %(stat,p_val))
        d = cohens_d(data,h_mean)
    else:
        # try a log transform if data positive
        s,p = shapiro(np.log(data))
        non_neg = np.all(data>0)
        if np.logical_and(p >= .05,non_neg):
            stat,p_val = ttest_1samp(np.log(data),np.log(h_mean),alternative=alt)
            print('......One-sample t-test,log-transformed data: '\
                  +'stat= %.2f, p-val = %.6f' %(stat,p_val))
            d = cohens_d(np.log(data),np.log(h_mean))
        else:
            # do non-parametric statistics
            stat,p_val =  wilcoxon(data-h_mean,alternative=alt)
            print('......Wicoxon test: stat= %.2f, p-val = %.6f' %(stat,p_val))
            d = cohens_d(data,h_mean)
    # d = cohens_d(data,h_mean)
    print('......Cohen''s d = %.2f' %d)


def run_contrast_paired_samples(data1,data2,alt='greater'):
    # check if normal distribution
    s,p = shapiro(data1-data2)      
    if p>=.05:
        # parametric stats
        stat,p_val = ttest_1samp(data1-data2,0,alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' %(stat,p_val))
        d = cohens_d(data1-data2,[0])
    else:
        # try log-transform
        non_neg = np.all(data1-data2>0)
        s,p = shapiro(np.log(data1-data2))
        if np.logical_and(p >= .05,non_neg):
            stat,p_val = ttest_1samp(np.log(data1-data2),0,alternative=alt)
            print('......One-sample t-test,log-transformed data: '\
                  +'stat= %.2f, p-val = %.6f' %(stat,p_val))
            d = cohens_d(np.log(data1-data2),[0])
        else:
            # do non-parametric statistics
            stat,p_val =  wilcoxon(data1-data2,alternative=alt)
            print('......Wicoxon test: stat= %.2f, p-val = %.6f' %(stat,p_val))        
            d = cohens_d(data1-data2,[0])
    print('......Cohen''s d = %.2f' %d)         


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
    d = cohens_d(data1,data2)
    print('......Cohen''s d = %.2f' %d)