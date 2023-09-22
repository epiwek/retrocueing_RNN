#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:28:03 2022

This script contains custom statistical functions and wrappers.

@author: emilia
"""
import numpy as np
from scipy.stats import shapiro, ttest_ind, ttest_1samp, mannwhitneyu, wilcoxon, chi2, pearsonr



def get_sem(data, dim=0):
    """
    Calculate the SEM of a dataset alongside the given dimension.
    :param dim: dimension alongside which to calculate the SEM
    :param data: data array
    :return:
    """
    n = data.shape[dim]
    sem = np.std(data, dim) / np.sqrt(n)
    return sem


def cohens_d(data1, data2):
    """
    Calculate Cohen's d statistic for a pair of datasets.
    :param np.ndarray data1: First dataset.
    :param np.ndarray or float or int data2: Second dataset or a hypothesised mean value.
    :return: Cohen's d
    """
    # calculate the size of the datasets
    n1, n2 = len(data1), 1 if (isinstance(data2, float) or isinstance(data2, int)) else len(data2)
    # calculate the variance of the samples
    s1 = np.var(data1, ddof=1)
    s2 = 1 if n2 == 1 else np.var(data2, ddof=1)  # set to 1 if data2 has only 1 point
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(data1), np.mean(data2)
    # calculate the effect size
    return (u1 - u2) / s


def corrcl(circ_var, l_var):
    """
    Calculate the correlation coefficient between a circular and linear variable.
    Removes NaNs from the dataset.

    Parameters
    ----------
    circ_var : array (n_samples,)
        vector containing the circular variable entries in radians.
    l_var : array (n_samples,)
        vector containing the linear variable entries.

    Returns
    -------
    rcl : float
        Correlation coefficient.
    p_val : float
        Probability value.

    """
    # get rid of nans
    nan_ix = np.where(np.isnan(l_var))[0]
    clean_ix = np.setdiff1d(np.arange(len(circ_var)), nan_ix)

    # get the sin and cos of circular samples
    sines = np.sin(circ_var[clean_ix])
    cosines = np.cos(circ_var[clean_ix])

    # calculate the partial correlation coefficient
    rcx, p1 = pearsonr(cosines, l_var[clean_ix])
    rsx, p2 = pearsonr(sines, l_var[clean_ix])
    rcs, p3 = pearsonr(sines, cosines)

    # calculate the full correlation coefficient
    rcl = np.sqrt((rcx ** 2 + rsx ** 2 - 2 * rcx * rsx * rcs) / (1 - rcs ** 2))

    # calculate the test statistic and check significance value
    test_stat = len(clean_ix) * rcl ** 2
    df = 2
    p_val = 1 - chi2.cdf(test_stat, df)

    return rcl, p_val


def run_contrast_single_sample(data, h_mean, alt='greater', try_log_transform=False):
    """
    Run a contrast for a single sample. Checks the distribution of the data and runs a one-sample t-test (if the
        distribution is not significantly different from normal), or a Wilcoxon test (if it is). Prints the statistic
        and p-value. For normally distributed data, also calculates and prints an effect size statistic (Cohen's d).
    :param np.array data: Data array.
    :param float h_mean: hypothesised mean value.
    :param str alt: Optional. Alternative hypothesis direction. Choose from 'greater', 'less', and 'two-sided'. Default
        is 'greater'.
    :param bool try_log_transform: Optional. If True and data is not normally distributed, checks if transforming it by
        taking a log will change the distribution to normal and if so, runs parametric statics on the transformed data.
        Default is False.

    """
    # check if normal distribution
    s, p = shapiro(data)
    if p >= .05:
        # parametric statistics
        stat, p_val = ttest_1samp(data, h_mean, alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' % (stat, p_val))
        d = cohens_d(data, h_mean)
        print('......Cohen''s d = %.2f' % d)
    else:
        if try_log_transform:
            # try a log transform if data positive
            s, p = shapiro(np.log(data))
            non_neg = np.all(data > 0)
            if np.logical_and(p >= .05, non_neg):
                stat, p_val = ttest_1samp(np.log(data), np.log(h_mean), alternative=alt)
                print('......One-sample t-test,log-transformed data: ' + 'stat= %.2f, p-val = %.6f' % (stat, p_val))
                d = cohens_d(np.log(data), np.log(h_mean))
                print('......Cohen''s d = %.2f' % d)
        else:
            # do non-parametric statistics
            stat, p_val = wilcoxon(data - h_mean, alternative=alt)
            print('......Wilcoxon test: stat= %.2f, p-val = %.6f' % (stat, p_val))
            print('Non-parametric effect sizes not implemented - use JASP.')


def run_contrast_paired_samples(data1, data2, alt='greater', try_log_transform=False):
    """
    Run a contrast for paired samples. Checks the distribution of the difference dataset (data1 - data2) and runs a
    one-sample t-test (if the  distribution is not significantly different from normal), or a Wilcoxon test (if it is).
    Prints the statistic and p-value. For normally distributed data, also calculates and prints an effect size statistic
    (Cohen's d).

    :param np.array data1: First data array.
    :param np.array data2: Second data array.
    :param str alt: Optional. Alternative hypothesis direction. Choose from 'greater', 'less', and 'two-sided'. Default
        is 'greater'.
    :param bool try_log_transform: Optional. If True and the difference dataset is not normally distributed, checks if
        transforming it by taking a log will change the distribution to normal and if so, runs parametric statics on the
        transformed data. Default is False.

    """
    # check if normal distribution
    s, p = shapiro(data1 - data2)
    if p >= .05:
        # parametric stats
        stat, p_val = ttest_1samp(data1 - data2, 0, alternative=alt)
        print('......One-sample t-test: stat= %.2f, p-val = %.6f' % (stat, p_val))
        d = cohens_d(data1 - data2, 0)
        print('......Cohen''s d = %.2f' % d)
    else:
        if try_log_transform:
            # try log-transform
            non_neg = np.all(data1 - data2 > 0)
            s, p = shapiro(np.log(data1 - data2))
            if np.logical_and(p >= .05, non_neg):
                stat, p_val = ttest_1samp(np.log(data1 - data2), 0, alternative=alt)
                print('......One-sample t-test,log-transformed data: stat= %.2f, p-val = %.6f' % (stat, p_val))
                d = cohens_d(np.log(data1 - data2), 0)
                print('......Cohen''s d = %.2f' % d)
        else:
            # do non-parametric statistics
            stat, p_val = wilcoxon(data1 - data2, alternative=alt)
            print('......Wilcoxon test: stat= %.2f, p-val = %.6f' % (stat, p_val))
            print('Non-parametric effect sizes not implemented.')


def run_contrast_unpaired_samples(data1, data2, alt='greater', try_log_transform=False):
    """
    Run a contrast for unpaired samples. Checks the distribution of both datasets and runs an independent-samples t-test
    (if both distributions are not significantly different from normal), or a mannwhitneyu test (if it is). Prints the
    statistic and p-value. For normally distributed data, also calculates and prints an effect size statistic (Cohen's
    d).
    :param np.array data1: First data array.
    :param np.array data2: Second data array.
    :param str alt: Optional. Alternative hypothesis direction. Choose from 'greater', 'less', and 'two-sided'. Default
        is 'greater'.
    :param bool try_log_transform: Optional. If True and data are not normally distributed, checks if transforming them
        by taking a log will change the distribution to normal and if so, runs parametric statics on the transformed
        data. Default is False.

    """
    # check if normal distribution
    s1, p1 = shapiro(data1)
    s2, p2 = shapiro(data2)

    if np.logical_and(p1 >= .05, p2 >= .05):
        # parametric stats
        stat, p_val = ttest_ind(data1, data2, alternative=alt)
        print('......Indiv. samples t-test: stat= %.2f, p-val = %.6f' % (stat, p_val))
    else:
        if try_log_transform:
            # try log-transform
            non_neg = np.logical_and(np.all(data1 > 0), np.all(data1 > 0))
            s1, p1 = shapiro(np.log(data1))
            s2, p2 = shapiro(np.log(data2))
            if np.all((p1 >= .05, p2 >= .05, non_neg)):
                stat, p_val = ttest_ind(np.log(data1), np.log(data2), alternative=alt)
                print('......Indiv. samples t-test,log-transformed data: stat= %.2f, p-val = %.6f' % (stat, p_val))
                d = cohens_d(data1, data2)
                print('......Cohen''s d = %.2f' % d)
        else:
            # do non-parametric statistics
            stat, p_val = mannwhitneyu(data1, data2, alternative=alt)
            print('......Mannwhitneyu test: stat= %.2f, p-val = %.6f' % (stat, p_val))
            print('Non-parametric effect sizes not implemented.')
