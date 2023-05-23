#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:56:15 2021

@author: emilia
"""
l=track_training['loss'].shape[-1]
window=constants.PARAMS['conv_criterion']['trial_window']
dLoss_all = torch.empty((l-window,))
loss_clean_all = torch.empty((l-window,))
conv = torch.empty((l-window,))
for i in range(window,l):
    dLoss,loss_clean, loss_mean = retnet.get_dLoss_dt(constants.PARAMS,track_training['loss'][:,i-window+1:i+1])
    dLoss_all[i-window] = dLoss[-1].clone()
    loss_clean_all[i-window] = loss_clean[-1].clone()
    conv[i-window]=retnet.apply_conv_criterion(constants.PARAMS,track_training['loss'][:,i-window+1:i+1])

plt.figure()
plt.plot(track_training['loss'][:,-(l-window):].mean(0),label='loss raw')
plt.plot(loss_clean_all,label='loss clean')
plt.plot(range(l-window),np.ones((l-window,))*constants.PARAMS['conv_criterion']['cond2'],'k--',label='threshold')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure()
plt.plot(dLoss_all)
plt.plot(range(len(dLoss_all)),np.ones((len(dLoss_all),))*constants.PARAMS['conv_criterion']['cond1'],'k--',label='threshold')
plt.tight_layout()
plt.ylabel('dLoss/dt')
plt.xlabel('epoch')



def get_dLoss(loss_vals,dt):
    # calculate the derivative
    loss_mean = loss_vals.mean(0)
    n_steps = loss_mean.shape[0]-dt
    dLoss = torch.zeros((n_steps,))
    for i in range(n_steps):
            dLoss[i] = loss_mean[i+dt] - loss_mean[i]
    return dLoss, loss_mean


sd = constants.PARAMS['conv_criterion']['smooth_sd']
sd = 10
dLoss_all_cleaned = gaussian_filter1d(dLoss_all,sd)


loss_clean_2 = torch.empty((l-window,))
for i in range(window,l):
    loss_clean_2[i-window] = track_training['loss'][:,i-window+1:i+1].mean()
    
    
p1 = np.polyfit(range(l-window),track_training['loss'][:,-(l-window):].mean(0),1)

y1 = np.polyval(p1,range(l-window))


p1 = np.polyfit(range(window),track_training['loss'][:,-window:].mean(0),1)
y1 = np.polyval(p1,range(window))

p1 = np.polyfit(range(l-window),dLoss_all,15)
dl1 = np.polyval(p1,range(l-window))
plt.plot(dl1)



p1 = np.polyfit(range(window),dLoss_all[-window:],1)
dl1 = np.polyval(p1,range(window))

