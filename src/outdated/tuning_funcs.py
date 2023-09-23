#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:19:08 2021

@author: emilia
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

n_stim = 4
n_colCh = 4

G = 0.5
kappa = 1.0




phi  = torch.linspace(0, 2*np.pi, n_colCh+1)[:-1] # # Tuning curve centers 
colour_space = torch.linspace(0, 2*np.pi, n_stim+1)[:-1] # possible colours
x = torch.linspace(0,2*np.pi,36)


# response to all possible stimuli for an example unit
# y = G * torch.exp(kappa * torch.cos(2 * (x - phi[0]))-1)
# y = G * torch.exp(kappa * torch.cos(x - phi[0]))/np.exp(kappa)
# y = G * torch.exp(kappa * torch.cos(x - phi[0])-1.0)


#%% individual to gauge width easier

plt.figure()
# y = G * torch.exp(kappa * torch.cos(x - phi[0])-2.0)


# old stimuli - messed up function
# y = G * torch.exp(kappa * torch.cos(x - phi[2]))/np.exp(kappa)

#WJM implementation
# y = G * torch.exp(kappa * torch.cos(2 * (x - phi[0]))-1)

# y = G * torch.exp(kappa * torch.cos(2*(x - phi[0])))


#XJW implementation
y = G * torch.exp(-(x - phi[2])**2/(2*kappa**2))

plt.plot(x,y,'k-')
plt.ylabel('unit activity')
plt.xlabel('colour [°]')    
plt.title('Colour tuning for an example input unit')
xx = np.linspace(0,360,7)
plt.xticks(ticks = np.radians(xx), labels=xx)

#%% all basis functions together
# plot
plt.figure()
for i in range(len(phi)):
    y = G * torch.exp(-(x - phi[i])**2/(2*kappa**2))
    # y = G * torch.exp(kappa * torch.cos(x - phi[i]))/np.exp(kappa)
    # y = y = G * torch.exp(kappa * torch.cos(2 * (x - phi[i]))-1)
    
    
    # y = y = G * torch.exp(kappa * torch.cos(2 * (x - phi[i])))
    plt.plot(x,y,'k-')

xx = np.linspace(0,360,7)
plt.xticks(ticks = np.radians(xx), labels=xx)

plt.ylabel('unit activity')
plt.xlabel('colour [°]')    
plt.title('Colour tuning curves, kappa = ' + str(kappa))


#%% von mises
from scipy.stats import vonmises
kappas = np.array([0.01,0.1,1,10,100,500])
x= np.linspace(-np.pi,np.pi,100)

fig_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/'\
            +'shared_repo/retrocue_rnn/emilia/vonMises_plots/'

fig,axs = plt.subplots(2,3,sharex=True, sharey=False)
# for k,kappa in enumerate(kappas): 
# # x = np.linspace(vonmises.ppf(0.01, kappa),
# #                  vonmises.ppf(0.99, kappa), 100)

    
#     ax[k].plot(x, vonmises.pdf(x, kappa,loc=0),
#             'r-', lw=2, alpha=0.6, label='vonmises pdf')
#     ax[k].title('kappa = '+str(kappa))
    
#     # if kappa < 1.0:
        
#     #     plt.savefig(fig_path+'kappa'+'1e'+str(np.int(np.log10(kappa))))
#     # else:
#     #     plt.savefig(fig_path+'kappa'+str(int(kappa)))
    
    
for i,ax in enumerate(axs.flat): 
# x = np.linspace(vonmises.ppf(0.01, kappa),
#                  vonmises.ppf(0.99, kappa), 100)

    kappa = kappas[i]
    # y = vonmises.pdf(x, kappa,loc=0)
    y = vonmises.pdf(x, kappa,loc=0)/max(vonmises.pdf(x, kappa,loc=0))
    
    ax.plot(x,y,  'r-', lw=2, alpha=0.6, label='vonmises pdf')
    ax.set_title('kappa = '+str(kappa))
    
    # if kappa < 1.0:
        
    #     plt.savefig(fig_path+'kappa'+'1e'+str(np.int(np.log10(kappa))))
    # else:
    #     plt.savefig(fig_path+'kappa'+str(int(kappa)))
    
    

# plt.ylim([0,1])
plt.tight_layout()
plt.xticks(np.arange(-2*np.pi,3*np.pi,np.pi),['-2π','-π','0','π','2π'])
# ax.set_xticks()
# ax.set_xticklabels()
