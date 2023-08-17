#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:34:57 2020

@author: emilia
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle
import pycircstat



plt.rcParams.update({'font.size': 13})

# models = np.array([ 0,  1,  2,  3,  5,  6,  8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
#        21, 23, 24, 25, 26, 27, 28, 29])


# models=range(10)
# models = np.array([0, 1, 2, 5, 7, 9])
# models = np.array(range(10))
# models = np.array([2, 3, 4, 5, 6, 7, 8, 9])



model_type = 'RNN'
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/pca_data/'
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/pca_data/'
#path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data/'
#path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data/'

# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian/with_fixation/nrec300/lr0.005/pca_data/'
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian_hardcoded/nrec300/lr0.005/pca_data/'
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_vonMises/MSELoss/kappa20.0/nrec200/lr0.005/pca_data/'
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot_new/nrec200/lr0.005/pca_data/'
path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/pca_data/'


f = open(path+'converged.pckl','rb')
models = pickle.load(f)
f.close()
n_models = len(models)


angles = np.zeros((n_models,2))
angles_radians = np.zeros((n_models,2))



plt.figure()

# for model in np.arange(n_models):
for i,model in enumerate(models):
    # load data
    f = open(path+ 'angles/angles_' + model_type + str(model)+ '.pckl','rb')
    obj = pickle.load(f)
    theta_pre, theta_post = obj
    f.close()
    angles[i,:] = [theta_pre,theta_post]
    angles_radians[i,:] = np.radians([theta_pre,theta_post])
    
    plt.plot(angles[i,:],'ko-',label='model '+str(model))

plt.xticks([0,1],labels=['pre-cue','post-cue'])
plt.xlim([-0.5,1.5])
plt.ylabel('Angle [degrees]')

# plt.savefig(path+model_type+'angles_fig')

#%% PVE by first 3 PCs in delay1 and 2
plt.figure()
pves = np.zeros((3,n_models,2)) # pre and post-cue
for i, model in enumerate(models):
    f = open(path+'/pca3/pca3_' + model_type + str(model)+'.pckl','rb')
    [delay1_pca, delay2_pca] = pickle.load(f)
    pves[:,i,0] = delay1_pca.explained_variance_ratio_
    pves[:,i,1] = delay2_pca.explained_variance_ratio_
    
    plt.plot(np.sum(pves[:,i,:],0),'ko-')


# plt.ylim([0.8,1.0])

# plt.yticks(np.arange(0.8,1.01,0.05))
plt.ylabel('PVE by first 3 PCs')

plt.xlim((-0.5,1.5))
plt.xticks([0,1],['pre-cue','post-cue'])

plt.plot(np.mean(np.sum(pves,0),0),'_-r')   
plt.plot(np.mean(np.sum(pves,0),0),marker=1,c='r')
plt.plot(np.mean(np.sum(pves,0),0),marker=0,c='r')    

 
#%% polar plot of angles

import seaborn as sns
plt.rcParams.update({'font.size': 30})

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,polar=True)
ax.grid(False)
r = 1
ms = 16
for i in range(n_models):
    
    ax.plot(angles_radians[i,:],np.ones((2,))*r,'k-',alpha=0.2)
    ax.plot(angles_radians[i,0],r,'o',color = cols[0],alpha=0.2,markersize=ms)
    ax.plot(angles_radians[i,1],r,'o',color = cols[1],alpha=0.2,markersize=ms)
    

ax.plot(pycircstat.descriptive.median(angles_radians[:,0]),r,'o',c = cols[0],markersize=ms,label='pre')
ax.plot(pycircstat.descriptive.median(angles_radians[:,1]),r,'o',c = cols[1],markersize=ms,label='post')




plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)

#%%
ax.set_yticks([])
ax.tick_params(axis='x', which='major', pad=23)

#%% circ stats t4ests


# get descriptive stats
mean_pre = pycircstat.descriptive.mean(angles_radians[:,0])
ci_pre = pycircstat.descriptive.mean_ci_limits(angles_radians[:,0], ci=.95)

mean_post = pycircstat.descriptive.mean(angles_radians[:,1])
ci_post = pycircstat.descriptive.mean_ci_limits(angles_radians[:,1], ci=.95)

# test for non-uniform distribution around 90
p_pre, v_pre = pycircstat.tests.vtest(angles_radians[:,0], np.radians(90))


print('V-test for uniformity/mean=90 :')
print('    v-stat = %.3f, p = %.3f' %(v_pre,p_pre))

# test for non-uniform distribution around 0

p_post, v_post = pycircstat.tests.vtest(angles_radians[:,1], 0)
print('V-test for uniformity/mean=0 :')
print('    v-stat = %.3f, p = %.3f' %(v_post,p_post))

# test for a significant difference in angles
angle_diff = angles_radians[:,0] - angles_radians[:,1]

diff = pycircstat.tests.mtest(angle_diff,0)
diff_mean = np.degrees(diff[1])
diff_result = diff[0]
diff_CI = (np.degrees(diff[2][1])-np.degrees(diff[2][0]))/2
# diff_SEM = (np.diff(diff_CI)/2)/1.96

print('circular one-sample t-test for angular difference ~=0 :')
print('     H = %d, mean = %.3f, CI = %.3f' %(diff_result[0],diff_mean,diff_CI))


#%% print descriptives

print('pre-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_pre),np.degrees(ci_pre)))
print('post-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_post),np.degrees(ci_post)))


#%% angle difference

angle_diff = angles_radians[:,1] - angles_radians[:,0]

p_diff = pycircstat.tests.medtest(angle_diff, np.radians(90))

p_diff_abs = pycircstat.tests.medtest(np.abs(angle_diff), np.radians(90))

fig = plt.figure()
ax = fig.add_subplot(111,polar=True)
ax.grid(False)
ax.plot(angle_diff,np.ones(angle_diff.shape),'ko')

ax.set_yticks([])



