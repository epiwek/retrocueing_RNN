#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:53:48 2021

@author: emilia
"""

import pickle
import constants
import matplotlib.pyplot as plt
import seaborn as sns
import custom_plot as cplot

#%%


noise_times = ['probe','delays','probe_and_delays','all']
noise_names = ['probe','delays','probe and delays','all']


epsilon = 1.0



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6),
                        sharex=True,sharey=True)

# cols = sns.color_palette("mako")
cols = sns.color_palette("dark")

c1 = cols[2]
c2 = cols[3]


for ax,time,ttl in zip(axs.flat,noise_times,noise_names):
    
    # load data
    load_path = constants.PARAMS['BASE_PATH'] + \
        'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g/' +\
              'epsilon'+str(epsilon)+'/' + time + '/'
                      
    data = pickle.load(open(load_path+'mean_sem_err_data.pckl','rb'))
    
    # cplot.shadow_plot(ax,data['bin_centres'],[data['valid_mean'],data['valid_sem']],
    #       precalc=True,alpha = 0.3, color = 'g')
    # cplot.shadow_plot(ax,data['bin_centres'],[data['invalid_mean'],data['invalid_sem']],
    #       precalc=True,color = 'r', alpha = 0.1)
    
    ax.fill_between(data['bin_centres'],data['valid_mean']+data['valid_sem'],
                    data['valid_mean']-data['valid_sem'], alpha = 0.2, color = c1)
    ax.plot(data['bin_centres'],data['valid_mean'],color=c1,label='valid')
    
    ax.fill_between(data['bin_centres'],data['invalid_mean']+data['invalid_sem'],
                    data['invalid_mean']-data['invalid_sem'], alpha = 0.2, color = c2)
    ax.plot(data['bin_centres'],data['invalid_mean'],color=c2,label='invalid')
    
    ax.set_title(ttl,{'fontsize': 15,'fontweight':20},pad=2.0)
    

plt.legend()

fig.add_subplot(111, frameon=False)#,visible=False)
plt.xticks([])
plt.yticks([])
plt.xlabel("Angular error [°]",labelpad=40)
plt.ylabel("Density",labelpad = 40)    
plt.tight_layout()


sns.despine(right=True,top=True)#,trim=True, offset = 10)

# sns.set_context('paper')

sns.set_context('notebook')

fig_path = constants.PARAMS['BASE_PATH'] + \
        'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g/' +\
              'epsilon'+str(epsilon)+'/valid_vs_invalid.png'
plt.savefig(fig_path)
