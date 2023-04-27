#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:45:58 2020

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
from make_stimuli import make_stimuli

#%% define plotting functions

def plot_geometry(ax,Z,pca,plot_colours):
    
    # plot the parallelogram defined by colours at location 1
    ax.plot(np.append(Z[:c_specs['n_subcats'],0],Z[0,0]),
          np.append(Z[:c_specs['n_subcats'],1],Z[0,1]),
          np.append(Z[:c_specs['n_subcats'],2],Z[0,2]),'k-')
    ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
             c='k',label='loc1')
    ax.scatter(Z[:c_specs['n_subcats'],0],Z[:c_specs['n_subcats'],1],
             Z[:c_specs['n_subcats'],2],marker='o',s = 40,c=plot_colours)
  
    # repeat for loc 2
    ax.plot(np.append(Z[c_specs['n_subcats']:,0],Z[c_specs['n_subcats'],0]),
          np.append(Z[c_specs['n_subcats']:,1],Z[c_specs['n_subcats'],1]),
          np.append(Z[c_specs['n_subcats']:,2],Z[c_specs['n_subcats'],2]),'k-')
    ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
             c='k',label='loc2')
    ax.scatter(Z[c_specs['n_subcats']:,0],Z[c_specs['n_subcats']:,1],
             Z[c_specs['n_subcats']:,2],marker='s',s = 40,c=plot_colours)

    ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
    ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
    ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')

    ax.legend()

  
def plot_subspace(ax,Y_l,scale=1.0,fc='k',a=0.2):
    # plot the best-fit plane
    verts = np.array([-Y_l.components_[0,:],-Y_l.components_[1,:],
                   Y_l.components_[0,:],Y_l.components_[1,:]])
    verts *= scale
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))


#%% set up model

class RNN(nn.Module):
    def __init__(self, n_inp, n_rec, n_out):
        super(RNN, self).__init__()
        # input
        self.n_rec = n_rec # number of recurrent neurons
        self.n_inp = n_inp
        self.n_out = n_out

        self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
        
        # change to Kaiming initialisation
        self.Wrec.weight_ih_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_inp)*(np.sqrt(2)/np.sqrt(self.n_inp)))
        self.Wrec.weight_hh_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_rec)*(np.sqrt(2)/np.sqrt(self.n_rec)))
        self.Wrec.bias_ih_l0 = nn.Parameter(torch.zeros(self.n_rec))
        self.Wrec.bias_hh_l0 = nn.Parameter(torch.zeros(self.n_rec))
        
        # input and hidden
        self.out = nn.Linear(self.n_rec, self.n_out) # output layer
        
        #ignore (scratch code):
        #self.inp = nn.Linear(17, self.n_neurons) # input weights
        #self.Wrec = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons))  # recurrent weights    
         #self.Wrec.bias_ih_l0 = nn.Parameter(torch.ones(n_rec))
        #self.Wrec.bias_hh_l0 = nn.Parameter(torch.ones(n_rec))
        #self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*1e-2)
        #self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*1e-2)
        
    # def step(self, input_ext, hidden):
    #     o, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
    #     output = self.out(hidden)
    #     return output, hidden
    #     # ignore:
    #     #hidden = torch.relu(torch.matmul(self.Wrec, hidden.unsqueeze(-1)).squeeze() + self.inp(input_ext))
        
    def forward(self, inputs):
        """
        Run the RNN with input timecourses
        """
        # Initialize network state
        #hidden = torch.zeros((1,inputs.size(1), self.n_rec)) # 0s
        
        o, h_n = self.Wrec(inputs)
        # Run the input through the network - across time
        # for i in range(inputs.size(2)):
        #     output, hidden = self.step(inputs[:, :, i].T, hidden)
        # return output.squeeze(), hidden
        
        output = self.out(h_n)
        return output.squeeze(), h_n
    
#%%  Set up model and task

# task parameters

n_stim = 4#32 # n colour stimuli
n_colCh = 4#32 # n color neurons/location
tuning_params = [0.8, 2.0] #height and width of tuning curve
trial_timings = {}
trial_timings['stim_dur']=1
trial_timings['delay1_dur'] = 1
trial_timings['cue_dur']= 1
trial_timings['delay2_dur'] = 1
trial_timings['ITI_dur'] = 0
trial_timings['resp_dur'] = 0

n_inp = n_stim*2 + 3 #stimuli + cues + fixation

n_rec = 10 # n hidden units
model = RNN(n_inp,n_rec,n_stim)

# Set up SGD parameters
n_iter = 4000 # iterations of SGD
batch_size = 50
learning_rate = .01#1e-5
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Squared error loss function
#loss_fn = nn.MSELoss()

# negLL
#loss_fn = nn.NLLLoss()

# cross-entropy
loss_fn = nn.CrossEntropyLoss()

## Run model

# Placeholders
track_loss = torch.zeros(n_iter)
track_targets = torch.zeros((batch_size,n_iter))
track_outputs = torch.zeros((batch_size,n_stim,n_iter))

# plot initial weights
#plt.figure()
#plt.subplot(121)
#plt.imshow(model.Wrec.weight_ih_l0.detach().numpy())
#plt.colorbar()
#plt.title('w_ih0')

#plt.subplot(122)
#plt.imshow(model.Wrec.weight_hh_l0.detach().numpy())
#plt.title('w_hh0')
#plt.colorbar()
#plt.tight_layout(pad=2.0)

#%% train model

# Loop over iterations
for i in range(n_iter):
    if (i + 1) % 50 == 0: # print progress every 100 iterations
        print('%.2f%% iterations of SGD completed...loss = %.2f' % (100* (i + 1) / n_iter, loss))
    # Sample stimulus
    I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings,tuning_params)

    # Run model
    outputs, hidden = model(I)
    # Compute loss
    loss = loss_fn(outputs, T)#.type(torch.long))
    # Keep track of outputs and loss
    track_loss[i] = loss.item()
    track_targets[:,i] = T.detach()
    track_outputs[:,:,i] = outputs.detach()
    # Compute gradients
    optimizer.zero_grad()
    loss.backward()
    # Update weights
    optimizer.step()

# Plot the loss
fig, ax = plt.subplots()
ax.plot(track_loss)
ax.set_xlabel('iterations of stochastic gradient descent')
ax.set_ylabel('loss')
#ax.set_ylim(0,1)

# plot trained weights
#plt.figure()
#plt.subplot(121)
#plt.imshow(model.Wrec.weight_ih_l0.detach().numpy())
#plt.colorbar()
#plt.title('w_ih')

#plt.subplot(122)
#plt.imshow(model.Wrec.weight_hh_l0.detach().numpy())
#plt.colorbar()
#plt.title('w_hh')

#plt.tight_layout(pad=2.0)

#%% save model

#path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/saved_models/'
path = 'saved_models/'
torch.save(model,path+'model1')

torch.save(model.state_dict(),path+'model1_statedict')
#%% load model

model = torch.load(path+'model1')

#%% ###### PCA part ######

Time = sum(trial_timings.values()) # sequence length
colour_space = torch.linspace(0, 2*np.pi, n_stim+1)[:-1] # all possible stimulus values

batch = 200
# bin data for PCA 
n_bins = 4 # colour bins
n_stimperbin = n_stim//n_bins #n of stim colours/bin
S = n_bins*2 # colour bin x location combos
n_samples = batch//S # n trials per category

# to do: add a proper error
if ((batch % S)!=0):
    print('Error: n_samples must be an integer')

else:
    c_specs = {} # constraint specs
    c_specs['n_subcats'] = n_bins
    c_specs['n_stimperbin'] = n_stimperbin
    c_specs['n_samples'] = n_samples
    # these are used by the make_stimuli function to constrain the number of trials
    # e.g. to be equal for each bin
    
    # evaluate model
    model.eval()
    I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch,trial_timings,tuning_params,
                                     constraints = 'on',c_specs = c_specs)
    with torch.no_grad():
        output, h_n = model.Wrec(I)
    
    loc = loc[:,:,0].squeeze() # get rid of the time dimension

#%% sort data for PCA

trial_data = torch.reshape(output.permute(1,0,-1).unsqueeze(0),(S,n_samples,Time,n_rec)) 
# reshape into condition (bin x location) x trial x time x unit
trial_data = trial_data.permute(1,-1,0,2) # n_samples x N x S x T


# do PCA and plot all trial timepoints
trial_periods = ['stim_dur','delay1_dur','cue_dur','delay2_dur']
fig=plt.figure(figsize = (24,6))
plot_colours = ['b','g','y','r']

for ix, time in enumerate(trial_periods):
    # extract data from a given timepoint
    X = torch.mean(trial_data[:,:,:,ix],0).T
    X -= torch.mean(X) # subtract mean
  
    # run pca
    pca = PCA(n_components=3) # Initializes PCA
    Z = pca.fit_transform(X)

    # initialise plot
    ax = fig.add_subplot(1,4,ix+1, projection='3d')
    Axes3D.mouse_init(ax)
  
    # plot
    plot_geometry(ax,Z,pca,plot_colours)

    plt.title(trial_periods[ix][:-4])
  
#%% plot pre-cue #####


fig=plt.figure(figsize = (8,8))
plot_colours = ['b','g','y','r']

ix = 1 # delay1 data column

X = torch.mean(trial_data[:,:,:,ix],0).T
X -= torch.mean(X,1).unsqueeze(-1) # demean
  
# run pca
pca = PCA(n_components=3) # Initializes PCA
Z = pca.fit_transform(X)

# initialise plot
ax = fig.add_subplot(111, projection='3d')
Axes3D.mouse_init(ax)
  
# plot
plot_geometry(ax,Z,pca,plot_colours)
plt.title('pre-cue')

# check angle between best-fitting planes

pca_plane1 = PCA(n_components=2)
pca_plane2 = PCA(n_components=2)
Y_loc1 = pca_plane1.fit(Z[:c_specs['n_subcats'],:])
Y_loc2 = pca_plane2.fit(Z[c_specs['n_subcats']:,:])

cos_theta_pre = np.dot(np.cross(Y_loc1.components_[0,:],Y_loc1.components_[1,:]),
                   np.cross(Y_loc2.components_[0,:],Y_loc2.components_[1,:]))

print(cos_theta_pre)
print(np.degrees(np.arccos(cos_theta_pre)))

# plot best-fitting planes

plot_subspace(ax,Y_loc1,2.0,a=.2)
plot_subspace(ax,Y_loc2,2.0,a=.2)

# calculate euclidean distances between corresponding points

dist_pre = np.zeros(n_bins,)
for i in range(n_bins):
    dist_pre[i] = np.linalg.norm(Z[i,:]-Z[i+n_bins,:])  
  

#%% plot post-cue ########


fig=plt.figure(figsize = (8,8))
plot_colours = ['b','g','y','r']

ix = 3 # delay1 data column

X = torch.mean(trial_data[:,:,:,ix],0).T
X -= torch.mean(X,1).unsqueeze(-1) # demean
  
# run pca
pca = PCA(n_components=3) # Initializes PCA
Z = pca.fit_transform(X)

# initialise plot
ax = fig.add_subplot(111, projection='3d')
Axes3D.mouse_init(ax)
  
# plot
plot_geometry(ax,Z,pca,plot_colours)
plt.title('post-cue')



pca_plane1 = PCA(n_components=2)
pca_plane2 = PCA(n_components=2)
Y_loc1 = pca_plane1.fit(Z[:c_specs['n_subcats'],:])
Y_loc2 = pca_plane2.fit(Z[c_specs['n_subcats']:,:])

cos_theta_post = np.dot(np.cross(Y_loc1.components_[0,:],Y_loc1.components_[1,:]),
                   np.cross(Y_loc2.components_[0,:],Y_loc2.components_[1,:]))

print(cos_theta_post)
print(np.degrees(np.arccos(cos_theta_post)))

plot_subspace(ax,Y_loc1,8.0,a=.2)
plot_subspace(ax,Y_loc2,8.0,a=.2)


# calculate euclidean distances between corresponding points 

dist_post = np.zeros(n_bins,)
for i in range(n_bins):
    dist_post[i] = np.linalg.norm(Z[i,:]-Z[i+n_bins,:])
    
    
#%% Train multiple networks and look at stats

# train a bunch of networks - see how many converge
Time = sum(trial_timings.values())

end_loss = []
n_models = 25
plane_angle = {}
plane_angle['precue'] = np.zeros((n_models,))
plane_angle['postcue'] = np.zeros((n_models,))
plane_fit = {}
plane_fit['var'] = np.zeros((n_models,4))
plane_fit['mean_dist'] = np.zeros((n_models,2))

for m in range(n_models):
    # initialise model
    model = RNN(n_inp,n_rec,n_stim)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # cross-entropy
    loss_fn = nn.CrossEntropyLoss()
    print('running model %i/%i...' % (m+1, n_models))

    # train
    print('   training...')

    for i in range(n_iter):
          if (i + 1) % (n_iter//5) == 0: # print progress every 20% iterations
              print('       %.2f%% iterations of SGD completed...loss = %.2f' % (100* (i + 1) / n_iter, loss))
          # Sample stimulus
          I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings,tuning_params)
    
          # Run model
          outputs, hidden = model(I)
          # Compute loss
          loss = loss_fn(outputs, T.type(torch.long))
          # Keep track of outputs and loss
          #track_loss[i] = loss.item()
          #track_targets[(i * batch_size) : ((i+1) * batch_size)] = target.detach().numpy()
          #track_outputs[(i * batch_size) : ((i+1) * batch_size)] = outputs.detach().numpy()
          # Compute gradients
          optimizer.zero_grad()
          loss.backward()
          # Update weights
          optimizer.step()
          if (i==n_iter-1):
            end_loss.append(loss.item())
        
          ## PCA
    print('   PCA...')
        
    model.eval()
    # evaluate model - on training data
    I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch,trial_timings,
                                       tuning_params, constraints = 'on',c_specs = c_specs)
    
    
    with torch.no_grad():
        output, h_n = model.Wrec(I.permute(-1,1,0))
    loc = loc[:,:,0].squeeze() # get rid of the time dimension
    # sort data
    trial_data = torch.reshape(output.permute(1,0,-1).unsqueeze(0),(S,n_samples,Time,n_rec))
    trial_av = torch.mean(trial_data,1)
    trial_av = trial_av.permute(-1,0,1) # unit x category x time
    trial_data = trial_data.permute(1,-1,0,2) # n_samples x N x S x T
    # center data
    trial_av -= torch.mean(trial_av.reshape(n_rec,-1),1)[:,None,None]

    # pre-cue
    ix = 1 # delay1 data column
    X = torch.mean(trial_data[:,:,:,ix],0).T
    X -= torch.mean(X,1).unsqueeze(-1) # demean
    
    # run pca
    pca = PCA(n_components=3) # Initializes PCA
    Z = pca.fit_transform(X)

    pca_plane1 = PCA(n_components=2)
    pca_plane2 = PCA(n_components=2)
    Y_loc1 = pca_plane1.fit(Z[:c_specs['n_subcats'],:])
    Y_loc2 = pca_plane2.fit(Z[c_specs['n_subcats']:,:])

    plane_angle['precue'][m] = np.abs(np.degrees(np.dot(np.cross(Y_loc1.components_[0,:],Y_loc1.components_[1,:]),
                       np.cross(Y_loc2.components_[0,:],Y_loc2.components_[1,:]))))
      
    dist_pre = np.zeros(n_bins,)
    for i in range(n_bins):
        dist_pre[i] = np.linalg.norm(Z[i,:]-Z[i+n_bins,:])
    plane_fit['var'][m,0] = np.sum(Y_loc1.explained_variance_ratio_)
    plane_fit['var'][m,1] = np.sum(Y_loc2.explained_variance_ratio_)
    plane_fit['mean_dist'][m,0] = np.mean(dist_pre)
    
    # post-cue
    ix = 3 # delay2 data column
    X = torch.mean(trial_data[:,:,:,ix],0).T
    X -= torch.mean(X,1).unsqueeze(-1) # demean
    
    # run pca
    pca = PCA(n_components=3) # Initializes PCA
    Z = pca.fit_transform(X)
    
    pca_plane1 = PCA(n_components=2)
    pca_plane2 = PCA(n_components=2)
    Y_loc1 = pca_plane1.fit(Z[:c_specs['n_subcats'],:])
    Y_loc2 = pca_plane2.fit(Z[c_specs['n_subcats']:,:])
    
    plane_angle['postcue'][m] = np.abs(np.degrees(np.dot(np.cross(Y_loc1.components_[0,:],Y_loc1.components_[1,:]),
                       np.cross(Y_loc2.components_[0,:],Y_loc2.components_[1,:]))))
      
    dist_post = np.zeros(n_bins,)
    for i in range(n_bins):
        dist_post[i] = np.linalg.norm(Z[i,:]-Z[i+n_bins,:])
    plane_fit['var'][m,2] = np.sum(Y_loc1.explained_variance_ratio_)
    plane_fit['var'][m,3] = np.sum(Y_loc2.explained_variance_ratio_)
    plane_fit['mean_dist'][m,1] = np.mean(dist_post)
    print('   Done.')

#%% plots 

fig = plt.figure(figsize=(13,4))
plt.rcParams.update({'font.size': 13})

plt.subplot(131)
for m in range(n_models):
    plt.plot([0,1],[plane_angle['precue'][m],plane_angle['postcue'][m]],'-',color='darkgrey')
    plt.plot([0],plane_angle['precue'][m],'o',color = 'sandybrown')
    plt.plot([1],plane_angle['postcue'][m],'o',color = 'sandybrown')
    

plt.ylabel('Absolute angle between planes')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in plane configuration')
plt.rcParams.update({'font.size': 13})


plt.subplot(132)
for m in range(n_models):
    plt.plot([0,1],[np.mean(plane_fit['var'][m,0:2]),np.mean(plane_fit['var'][m,2:])],'-',color='darkgrey')
    plt.plot([0],np.mean(plane_fit['var'][m,0:2]),'o',color='lightcoral')
    plt.plot([1],np.mean(plane_fit['var'][m,2:]),'o',color='lightcoral')
   

plt.ylabel('Mean PVE by planes')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in plane fit')
plt.rcParams.update({'font.size': 13})



plt.subplot(133)
for m in range(n_models):
    plt.plot([0,1],plane_fit['mean_dist'][m,:],'-',color='darkgrey')
    plt.plot([0],plane_fit['mean_dist'][m,0],'o',color='palevioletred')
    plt.plot([1],plane_fit['mean_dist'][m,1],'o',color='palevioletred')
    

plt.ylabel('Mean point distance')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in point correspondence')
plt.rcParams.update({'font.size': 13})

fig.tight_layout(pad=2.0)

print('mean ± SEM plane angle:')
print('    pre-cue: %.2f ± %.2f' % 
      (np.mean(plane_angle['precue']),np.std(plane_angle['precue'])/np.sqrt(n_models)))
print('    post-cue: %.2f ± %.2f' % 
      (np.mean(plane_angle['postcue']),np.std(plane_angle['postcue'])/np.sqrt(n_models)))

# check the distribution is normal:
sw, sw_p = stats.shapiro(plane_angle['precue']-plane_angle['postcue'])

if sw_p > 0.05:
    t, p = stats.ttest_1samp(plane_angle['precue']-plane_angle['postcue'],0)
    print('test the difference in means:')
    if t>0: # expect the difference to be positive
        print('     t = %.2f \n     p = %.2f' % (t,p/2))
    else:
        print('     t = %.2f \n     p = %.2f' % (t,1-p/2))
else:
    print('data not normally distributed')
#%% IGNORE
#from matplotlib import cm
#from colorspacious import cspace_converter
#from collections import OrderedDict

#cmaps = OrderedDict()    
#cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
#                        'Dark2', 'Set1', 'Set2', 'Set3',
#                        'tab10', 'tab20', 'tab20b', 'tab20c']

#%% distribution of pre-post angle difference
plt.figure()
plt.hist((plane_angle['precue']-plane_angle['postcue']),weights=np.ones((n_models))/n_models,color='sandybrown')
plt.xlabel('plane angle pre - post cue')
plt.ylabel('proportion of models')

#%% plot example stimuli

batch_size = 2

I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings)
# I is channel x batch x time

fig, axs = plt.subplots(nrows=1, ncols = batch_size, sharex = True, sharey = True)
fig.set_size_inches(8,8)

for i,ax in enumerate(fig.axes):
    ax.imshow(I[:,i,:])
    if i==0:
        ax.set_xlabel('time')
        ax.set_ylabel('input channel')
        t = np.fromiter(trial_timings.values(),dtype=float)
        t = np.delete(t,np.where(t==0))
        t = np.cumsum(t)-1
        ax.set_xticks(t)
        ax.set_xticklabels(['S','D1','C','D2'])
        ax.set_yticks(np.arange(0,n_inp,3))

fig.colorbar(axs[i].imshow(I[:,i,:]))
fig.tight_layout()

#%% look at 'outlier' networks - where points no longer lie on a plane after cue

row, col = np.where(plane_fit['var']<0.95)

fig = plt.figure(figsize=(13,4))
plt.rcParams.update({'font.size': 13})

plt.subplot(131)
for m in range(n_models):
    plt.plot([0,1],[plane_angle['precue'][m],plane_angle['postcue'][m]],'-',color='darkgrey')
    if len((np.where(row==m))[0])>0:
        plt.plot([0],plane_angle['precue'][m],'^',color = 'k')
        plt.plot([1],plane_angle['postcue'][m],'^',color = 'k')
    else:
        plt.plot([0],plane_angle['precue'][m],'o',color = 'sandybrown')
        plt.plot([1],plane_angle['postcue'][m],'o',color = 'sandybrown')
    

plt.ylabel('Absolute angle between planes')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in plane configuration')
plt.rcParams.update({'font.size': 13})


plt.subplot(132)
for m in range(n_models):
    plt.plot([0,1],[np.mean(plane_fit['var'][m,0:2]),np.mean(plane_fit['var'][m,2:])],'-',color='darkgrey')
    if len((np.where(row==m))[0])>0:
        plt.plot([0],np.mean(plane_fit['var'][m,0:2]),'^',color='k')
        plt.plot([1],np.mean(plane_fit['var'][m,2:]),'^',color='k')
    else:
        plt.plot([0],np.mean(plane_fit['var'][m,0:2]),'o',color='lightcoral')
        plt.plot([1],np.mean(plane_fit['var'][m,2:]),'o',color='lightcoral')
   

plt.ylabel('Mean PVE by planes')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in plane fit')
plt.rcParams.update({'font.size': 13})



plt.subplot(133)
for m in range(n_models):
    plt.plot([0,1],plane_fit['mean_dist'][m,:],'-',color='darkgrey')
    if len((np.where(row==m))[0])>0:
        plt.plot([0],plane_fit['mean_dist'][m,0],'^',color='k')
        plt.plot([1],plane_fit['mean_dist'][m,1],'^',color='k')
    else:
        plt.plot([0],plane_fit['mean_dist'][m,0],'o',color='palevioletred')
        plt.plot([1],plane_fit['mean_dist'][m,1],'o',color='palevioletred')
    

plt.ylabel('Mean point distance')
plt.xlim((-0.2,1.2))
plt.xticks([0,1],['pre-cue','post-cue'])
plt.title('Change in point correspondence')
plt.rcParams.update({'font.size': 13})

fig.tight_layout(pad=2.0)



#%% IGNORE

# plt.close('all')
# timepoints = np.linspace(1,n_iter,4,dtype=int)-1
# fig = plt.figure(5,40)
# for ax,timepoint in enumerate(timepoints):
#     fig.add_subplot(1,8,(ax//4)+1)
#     tmp = torch.zeros(track_outputs[:,:,timepoint].shape)
#     for i in range(batch_size):
#         tmp[i,int(track_targets[i,timepoint])]=1
#     plt.imshow(tmp)#,vmin=0,vmax=1)
#     plt.title('targets')
    
#     fig.add_subplot(1,8,(ax//4)+2)
#     plt.imshow(sm(track_outputs[:,:,timepoint]),vmin=0,vmax=1)
#     plt.colorbar()
#     plt.title('outputs')
    
#     plt.tight_layout(pad=0)

