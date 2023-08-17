#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:03:51 2021

@author: emilia
"""
import scipy.optimize as opt
import numpy as np

x = np.array([0,0.01,0.05,0.1,0.3,0.5,0.7])
y = np.array([12,18,47,150,1500,3100,3400])/3400

def f(x, b, c, d):
    return 1 / (1. + np.exp(-c * (x - d))) + b

(b_, c_, d_), _ = opt.curve_fit(f, x, y)


y_fit = f(x, b_, c_, d_)



def f(x, b, c, d):
    return 1 / (1. + np.exp(-c * (x - d))) + b


# np.round((1 / (1 + np.exp(1 + np.exp(-13.6 * (x - 0.32)) - 0.007) * 340)
np.round((1 / (1. + np.exp(-13.589 * (x - 0.316))) - 0.007) * 340)



#%% 

import constants
path = constants.PARAMS['FULL_PATH']


# noise_lvls = [0,0.01,0.05,0.1,0.3,0.5]
noise_lvls = np.array([0,0.1,0.2,0.3,0.5,0.5])
noise_lvls = np.sqrt(noise_lvls**2 / len(constants.PARAMS['noise_timesteps']))


plateau_length_median = np.empty((len(noise_lvls)))
plateau_length = np.empty((len(noise_lvls),10))

for i,n in enumerate(noise_lvls):
    path = constants.PARAMS['COND_PATH'] \
        +'epsilon' + str(n)\
            +'/kappa' + str(constants.PARAMS['kappa_val'])\
            +'/nrec' + str(constants.PARAMS['n_rec'])\
                +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                
    for m in np.arange(constants.PARAMS['n_models']):
        model_number = str(m)
        
        # load training data
        f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        plateau_length[i,m] = np.logical_and(track_training['loss_epoch']<0.007,track_training['loss_epoch']>0.006).sum()
    
    plateau_length_median[i] = np.median(plateau_length[i,:])
    

plt.plot(noise_lvls,plateau_length_median,'k-o')

plt.plot(noise_lvls,np.sqrt(plateau_length_median)*5,'k-o')
# this looks like a good formula

#%%

import constants
import pickle
import seaborn as sns
path = constants.PARAMS['FULL_PATH']
plt.rcParams.update({'font.size': 10})


# noise_lvls = [0.01,0.03,0.05,0.07,0.1]#,0.1,0.5]
noise_lvls_target = np.array([0.1,0.2,0.3,0.5,0.7])


# noise_lvls = np.array([0,0.1,0.2,0.3])#,0.5,0.7])
noise_lvls = np.sqrt(noise_lvls_target**2 / len(constants.PARAMS['noise_timesteps']))

plateau_length_median = np.empty((len(noise_lvls)))
plateau_length = np.empty((len(noise_lvls),10))
plt.figure()
end_loss = np.empty((10,len(noise_lvls)))
for m in np.arange(constants.PARAMS['n_models']):
    model_number = str(m)
    # plt.figure(figsize=(9,5))
    plt.subplot(2,5,m+1)
    plt.title('Model ' + model_number)
    for i,n in enumerate(noise_lvls):
        path = constants.PARAMS['COND_PATH'] \
            +'epsilon' + str(n)\
                +'/kappa' + str(constants.PARAMS['kappa_val'])\
                +'/nrec' + str(constants.PARAMS['n_rec'])\
                    +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                    
    
        
        
        # load training data
        f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        plt.plot(track_training['loss_epoch'],c = colours[i,:],label=str(noise_lvls_target[i]))
        end_loss[m,i] = track_training['loss_epoch'][-1]
        # if n == 0.01:
        #     end_loss[m] = track_training['loss_epoch'][-1]
        # if n == 0.5:
        #     plt.plot(np.arange(len(track_training['loss_epoch'])),
        #              torch.ones((len(track_training['loss_epoch']),))*end_loss[m],'k--')
        #plateau_length[i,m] = np.logical_and(track_training['loss_epoch']<0.007,track_training['loss_epoch']>0.006).sum()
plt.legend(bbox_to_anchor = (1,1))
    #plateau_length_median[i] = np.median(plateau_length[i,:])
    
# plt.figure()
# for i,n in enumerate(noise_lvls):
#     plt.scatter(np.ones((10,))*n,end_loss[:,i],c=colours[i,:])
# plt.plot(noise_lvls,np.median(end_loss,0),"k_",label='median')

# plt.xlabel('noise level')
# plt.ylabel('end loss value')

# plt.legend()


# for i,n in enumerate(noise_lvls):
#         path = constants.PARAMS['COND_PATH'] \
#             +'epsilon' + str(n)\
#                 +'/kappa' + str(constants.PARAMS['kappa_val'])\
#                 +'/nrec' + str(constants.PARAMS['n_rec'])\
#                     +'/lr' + str(constants.PARAMS['learning_rate']) + '/saved_models/'
#     for m in np.arange(constants.PARAMS['n_models']):
#         model = retnet.load_model(path,constants.PARAMS,device = torch.device('cpu'))
#%%


import seaborn as sns
colours = np.array(sns.color_palette("husl", constants.PARAMS['n_models']))

device = torch.device('cpu')
model, RNN = retnet.define_model(constants.PARAMS, device)

        
load_path = constants.PARAMS['FULL_PATH'] + 'saved_models/'

# model = retnet.load_model(path,constants.PARAMS,device)
# model = torch.load(path+'model'+str(m))

noise_lvls_target = np.array([0.1,0.2,0.3,0.5,0.7])
# noise_lvls = np.array([0,0.1,0.2,0.3])#,0.5,0.7])
noise_lvls = np.sqrt(noise_lvls_target**2 / len(constants.PARAMS['noise_timesteps']))

inp_weight_stats = np.empty((10,len(noise_lvls),2))
hidden_weight_stats = np.empty((10,len(noise_lvls),2))
inp_bias_stats = np.empty((10,len(noise_lvls),2))

for i,n in enumerate(noise_lvls):
    path = constants.PARAMS['COND_PATH'] \
            +'epsilon' + str(n)\
                +'/kappa' + str(constants.PARAMS['kappa_val'])\
                +'/nrec' + str(constants.PARAMS['n_rec'])\
                    +'/lr' + str(constants.PARAMS['learning_rate']) + '/saved_models/'
    for m in np.arange(constants.PARAMS['n_models']):
        constants.PARAMS['model_number'] = m
        model = retnet.load_model(path,constants.PARAMS,device)
        
        inp_weight_stats[m,i,:] = torch.std_mean(model.Wrec.detach())
        inp_bias_stats[m,i,:] = torch.std_mean(model.inp.bias.detach())
        hidden_weight_stats[m,i,:] = torch.std_mean(model.inp.weight.detach())
        
plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,hidden_weight_stats[:,i,0],c=colours[i,:])
plt.ylabel('Hidden weight std')
plt.xlabel('noise level')
plt.tight_layout()


plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,hidden_weight_stats[:,i,1],c=colours[i,:])
plt.ylabel('Hidden weight mean')
plt.xlabel('noise level')
plt.tight_layout()

plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,inp_weight_stats[:,i,0],c=colours[i,:])
plt.ylabel('input weight std')
plt.xlabel('noise level')
plt.tight_layout()

plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,inp_weight_stats[:,i,1],c=colours[i,:])
plt.ylabel('input weight mean')
plt.xlabel('noise level')
plt.tight_layout()


plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,inp_bias_stats[:,i,1],c=colours[i,:])
plt.ylabel('input bias mean')
plt.xlabel('noise level')
plt.tight_layout()


plt.figure()
for i,n in enumerate(noise_lvls):
    plt.scatter(np.ones((10,))*n,inp_bias_stats[:,i,1],c=colours[i,:])
plt.ylabel('input bias mean')
plt.xlabel('noise level')
plt.tight_layout()



# quick and dirty regressions
# note for every level of noise, model number n is initialised with the same 
# weights and 'sees' the same sequence of trials (noise might even be sampled 
# in the same way, just rescaled - this is all because setting the seed)
# so the assumption of independence is violated between different noise levels

from scipy.stats import linregress
x_vals = np.stack([noise_lvls]*10).T.reshape(-1)
y_vals = hidden_weight_stats[:,:,0].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)

y_vals = hidden_weight_stats[:,:,1].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)

y_vals = inp_weight_stats[:,:,0].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)

y_vals = inp_weight_stats[:,:,1].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)

y_vals = inp_bias_stats[:,:,0].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)

y_vals = inp_bias_stats[:,:,1].T.reshape(-1)
slope, intercept, r_value, p_value, std_err = linregress(x_vals,y_vals)
print(p_value)


#%%

load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'


outputs = model_outputs['data']

hidden_activ_means = np.empty((10,len(noise_lvls),13))
for i,n in enumerate(noise_lvls):
    path = constants.PARAMS['COND_PATH'] \
            +'epsilon' + str(n)\
                +'/kappa' + str(constants.PARAMS['kappa_val'])\
                +'/nrec' + str(constants.PARAMS['n_rec'])\
                    +'/lr' + str(constants.PARAMS['learning_rate'])\
                        + '/pca_data/valid_trials/'
    for m in np.arange(constants.PARAMS['n_models']):
        constants.PARAMS['model_number'] = m
        
        activations = pickle.load(open(path+'pca_data_model'+str(m)+'.pckl','rb'))
        
        hidden_activ_means[m,i,:] = activations['data'].mean(-1).mean(0)
        