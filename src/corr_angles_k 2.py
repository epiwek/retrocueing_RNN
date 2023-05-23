#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:08:51 2022

@author: emilia
"""
import pickle
import numpy as np
import pandas as pd
import pycircstat
from scipy.stats import chi2, pearsonr, shapiro
import constants
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import get_subspace_alignment_index as ai


path1 = '/Volumes/EP_Passport/emilia/data_vonMises/MSELoss_custom/validity_paradigm/3delays/probabilistic/all/sigma0.0/kappa5.0/nrec200/lr0.0001/'
path2 = '/Volumes/EP_Passport/emilia/data_vonMises/MSELoss_custom/validity_paradigm/3delays/neutral/all/sigma0.0/kappa5.0/nrec200/lr0.0001/'
angles_p = pickle.load(open(path1+'pca_data/valid_trials/all_plane_angles.pckl','rb'))
angles_radians_p = np.radians(angles_p)

angles_n = pickle.load(open(path2+'pca_data/valid_trials/all_plane_angles.pckl','rb'))
angles_radians_n = np.radians(angles_n)

#%% is there a difference in post-probe angles between probabilistic and neutral conditions
#on valid trials?

p, T = pycircstat.tests.watson_williams(angles_radians_n[:,-1],angles_radians_p[:,-1])

#%% are the plane angles and precision parameters correlated across the two conditions?

# load parameter values

load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'

K_table = pd.read_csv(load_path + 'K_table.csv')

K_valid = K_table['K_valid'].to_numpy()
group = K_table['condition']




lb = LabelBinarizer()
group_1hot = lb.fit_transform(group).squeeze()


angles_all = np.concatenate((angles_radians_n[:,1],angles_radians_p[:,1]),0)

p_ix = np.where(group_1hot)[0]
n_ix = np.where(group_1hot==0)[0]


def corrcl(circ_var,l_var,group_1hot):
    # get rid of nans
    nan_ix = np.where(np.isnan(l_var))[0]
    clean_ix = np.setdiff1d(np.arange(len(circ_var)),nan_ix)
    
    # group_counts = []
    # g_ixs = []
    # for g in range(2):
    #     group_counts.append((group_1hot[clean_ix]==g).sum())
    #     g_ixs.append(clean_ix[group_1hot[clean_ix]==g])
    
    
    # rewrite this to do it multiple times
    # if len(np.unique(group_counts))>1:
    #     # sample equal group numbers
    #     n = np.min(group_counts)
        
    #     rng = np.random.default_rng()
    #     s1_ix = rng.choice(g_ixs[0],size=(n,),replace=False)
    #     s2_ix = rng.choice(g_ixs[1],size=(n,),replace=False)
    # else:
    # s1_ix = np.arange(g_ixs[0])
    # s2_ix = np.arange(g_ixs[1])
    # all_ixs = np.sort(np.concatenate((s1_ix,s2_ix),0))
    all_ixs = clean_ix
    sines = np.sin(circ_var[all_ixs])
    cosines = np.cos(circ_var[all_ixs])
    
        



    rcx,p1 = pearsonr(cosines,l_var[all_ixs])
    rsx,p2 = pearsonr(sines,l_var[all_ixs])
    rcs,p3 = pearsonr(sines,cosines)

    rcl = np.sqrt((rcx**2 + rsx**2 - 2*rcx*rsx*rcs)/(1-rcs**2))

    test_stat = len(all_ixs)*rcl**2
    
    

    df = 2
    
    p_val = 1 - chi2.cdf(test_stat,df)
    
    return rcl, p_val

rcl,p = corrcl(angles_all,K_valid,group_1hot)


plt.figure()
plt.plot(angles_all,K_valid,'ko')
plt.xlabel('post-probe angle')
plt.ylabel('K')

plt.figure()
plt.plot(angles_all,np.log(K_valid),'ko')
plt.xlabel('post-probe angle')
plt.ylabel('log(K)')


#%% is the AI correlated with K on valid trials


# ai_p = ai.get_AI_cued_within_delay(constants,path1+'pca_data/valid_trials/')
# ai_n = ai.get_AI_cued_within_delay(constants,path2+'pca_data/valid_trials/')
# ai_all = np.concatenate((ai_n[0,1,:].squeeze(),ai_p[0,1,:]))


ai_p = ai.get_AI_cued_probe(constants,path1+'pca_data/valid_trials/')
ai_n = ai.get_AI_cued_probe(constants,path2+'pca_data/valid_trials/')



ai_all = np.concatenate((ai_n[0,:].squeeze(),ai_p[0,:]))

plt.figure()
plt.hist(ai_all)

nan_ix = np.where(np.isnan(K_valid))[0]
clean_ix = np.setdiff1d(np.arange(len(ai_all)),nan_ix)


shapiro(ai_all[clean_ix])
shapiro(np.log(K_valid[clean_ix]))

# shapiro(np.sqrt(ai_all[clean_ix]))
# shapiro(np.log(K_valid[clean_ix]))

plt.figure()
# plt.plot(np.log(ai_all[clean_ix]),np.log(K_valid[clean_ix]),'ko')
# plt.ylabel('log(K)')
# plt.xlabel('log(AI)')

plt.plot(ai_all[clean_ix],np.log(K_valid[clean_ix]),'ko')

plt.ylabel('log(K)')
plt.xlabel('AI')



plt.figure()
plt.plot(ai_all[clean_ix],K_valid[clean_ix],'ko')
plt.ylabel('K')
plt.xlabel('AI')


pearsonr(ai_all[clean_ix],np.log(K_valid[clean_ix]))


#%% plot each condition separately
import seaborn as sns
cols = sns.color_palette("Set2")[2:0:-1]

plt.figure()
plt.plot(ai_all[n_ix],np.log(K_valid[n_ix]),'o',c=cols[0])
plt.ylabel('K')
plt.xlabel('log(AI)')
plt.title('neutral, valid trials')

n_clean_ix = np.intersect1d(n_ix,clean_ix)
pearsonr(np.log(ai_all[n_clean_ix]),np.log(K_valid[n_clean_ix]))


plt.figure()
plt.plot(ai_all[p_ix],np.log(K_valid[p_ix]),'o',c=cols[1])
plt.ylabel('log(K)')
plt.xlabel('AI')
plt.title('probabilistic, valid trials')

p_clean_ix = np.intersect1d(p_ix,clean_ix)
pearsonr(ai_all[p_clean_ix],np.log(K_valid[p_clean_ix]))



#%% look at valid vs invalid trials - AI in probabilistic condition

ai_p = ai.get_AI_cued_probe(constants,path1+'pca_data/invalid_trials/')
K_invalid = K_table['K_invalid'].to_numpy()

plt.figure()
plt.plot(ai_p[0,:],np.log(K_invalid[p_ix]),'o',c=cols[1])
plt.ylabel('log(K)')
plt.xlabel('AI')
plt.title('probabilistic, invalid trials')

plt.figure()
plt.plot(np.ones(len(p_ix)),ai_p[0,:],'o',c=cols[1])
plt.ylim([0,1])

pearsonr(ai_p[0,:],np.log(K_valid[p_clean_ix]))


#%% does the angle between probed and unprobed in post-probe delay predict performance?

ai_p = ai.get_AI_cued_vs_uncued_probe(constants,path1+'pca_data/valid_trials/')
ai_n = ai.get_AI_cued_vs_uncued_probe(constants,path2+'pca_data/valid_trials/')

ai_all = np.concatenate((ai_n[0,:].squeeze(),ai_p[0,:]))


plt.figure()
plt.hist(ai_all)

nan_ix = np.where(np.isnan(K_valid))[0]
clean_ix = np.setdiff1d(np.arange(len(ai_all)),nan_ix)


shapiro(np.log(ai_all[clean_ix]))
shapiro(np.log(K_valid[clean_ix]))


plt.figure()
plt.plot(np.log(ai_all[clean_ix]),np.log(K_valid[clean_ix]),'ko')
plt.ylabel('log(K)')
plt.xlabel('log(AI)')

pearsonr(np.log(ai_all[clean_ix]),np.log(K_valid[clean_ix]))


#%% regression - which of the geometries predicts performance best
 
# buschman networks
AI_cued = ai.get_AI_cued_within_delay(constants)
AI_uncued = ai.get_AI_uncued_within_delay(constants)
AI_cu = ai.get_AI_cued_vs_uncued(constants)  

path = constants.PARAMS['FULL_PATH']

X = np.stack((AI_cued[0,1,:],AI_uncued[0,1,:],AI_cu[0,1,:]))

#%%
max_n_trials = 160
all_loss = np.zeros((constants.PARAMS['n_models'],max_n_trials))
n_epochs_to_convergence = []
for model in range(constants.PARAMS['n_models']):
    model_number = str(model)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    
    n_epochs_to_convergence.append(len(track_training['loss_epoch']))
    
n_epochs_to_convergence = np.array(n_epochs_to_convergence)

# area under the curve - cummulative loss

cum_loss = all_loss.sum(-1)

# put into a table and export to JASP

tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],X.T),axis=1)

df = pd.DataFrame(data=tbl,columns=['n epochs','cum_loss','cued AI', 'uncued AI', 'cued-uncued AI'])
df.to_csv(constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/reg_tbl.csv')
