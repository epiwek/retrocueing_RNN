#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:18:41 2020

@author: emilia
"""

# from pickle import TRUE
import numpy as np
#import numpy.random as rd
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats

XaviersComputer = False

# if not(XaviersComputer):
#     cd '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'
    
from utils import generate_data
import vec_operations as vops


if XaviersComputer:
    LSTM_PATH = '/Users/Xavier/Desktop/Work/Summerfield-Lab/retrocue_rnn/xavier-timo/src/pytorch-weights.npy'
    DENSE_PATH = '/Users/Xavier/Desktop/Work/Summerfield-Lab/retrocue_rnn/xavier-timo/src/Dense_weights.npy'
else:
    LSTM_PATH = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/xavier-timo/src/pytorch-weights.npy'
    DENSE_PATH = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/xavier-timo/src/Dense_weights.npy'
    

# load weights and data
x, y, lab = generate_data(labels=True)

x_data = torch.from_numpy(x)
y_data = torch.from_numpy(y)

x_data = x_data.type(torch.float32)
y_data = y_data.type(torch.float32)

lstm_weights = np.load(LSTM_PATH ,allow_pickle=True)
dense_weights = np.load(DENSE_PATH,allow_pickle=True)

x_data = x_data.permute(1,0,-1)

odds = x_data[:,0:32:2,:]
evens_reds = x_data[:,1:32:8,:]
evens_blues = x_data[:,3:32:8,:]
evens_greens = x_data[:,5:32:8,:]
evens_yellows = x_data[:,7:32:8,:]
x_data = torch.cat((odds, evens_reds, evens_blues, evens_greens, evens_yellows), 1)


#%% define functions

def plot_geometry(ax,Z,pca,plot_colours,plot_outline = True):
    
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(Z[:n_colours,0],Z[0,0]),
              np.append(Z[:n_colours,1],Z[0,1]),
              np.append(Z[:n_colours,2],Z[0,2]),'k-')
    ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
              c='k',label='loc1')
    ax.scatter(Z[:n_colours,0],Z[:n_colours,1],
              Z[:n_colours,2],marker='o',s = 40,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(Z[n_colours:,0],Z[n_colours,0]),
              np.append(Z[n_colours:,1],Z[n_colours,1]),
              np.append(Z[n_colours:,2],Z[n_colours,2]),'k-')
    ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
              c='k',label='loc2')
    ax.scatter(Z[n_colours:,0],Z[n_colours:,1],
              Z[n_colours:,2],marker='s',s = 40,c=plot_colours)

    ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
    ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
    ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')

    ax.legend()

def plot_plane(ax,verts,fc='k',a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sortByPathLength(verts)
    
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts


def plot_plane_old(ax,Y_l,points,scale=1.0,fc='k',a=0.2):
    # plot the best fitting plane as a quadrilateral, with vertices being some
    #scaled distance from the centre-of-mass of the original point set
    # (plots look better if e.g. one set of points forms a much denser cloud than the other)

    com = np.mean(points,axis=0) # centre of mass
    #sorted_verts, sorting_order = vops.sortByPathLength(points)
    # set the scale to be equal to largest distance between points
    #scale = np.linalg.norm(sorted_verts[0]-sorted_verts[2])
    Y_l.components_ *= scale
    # plot the best-fit plane
    verts = np.array([com-Y_l.components_[0,:],com-Y_l.components_[1,:],
                    com+Y_l.components_[0,:],com+Y_l.components_[1,:]])
    #verts *= scale
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    

#%% define model
torch.manual_seed(0)

class RNN(nn.Module):
    def __init__(self, n_inp, n_rec, n_out):
        super(RNN, self).__init__()
        # input
        self.n_rec = n_rec # number of recurrent neurons
        self.n_inp = n_inp
        self.n_out = n_out
        self.lstm = nn.LSTM(n_inp,n_rec)
        
        #self.out = nn.Linear(self.n_rec, self.n_out) # output layer
        
    #def step(self, input_ext, hidden, cell):
    #    o, (hidden, cell) = self.lstm(input_ext.unsqueeze(0),(hidden, cell))
    #    output = self.out(o)
    #    return output, hidden, cell
        
    def forward(self, inputs):
        """
        Run the RNN with input timecourses
        """
        # Initialize network state
        #h_0 = torch.zeros((1,inputs.size(1), self.n_rec)) # n layers x batch x n_rec
        #c_0 = torch.zeros((1,inputs.size(1), self.n_rec))
        # Run the input through the network - across time
        #for i in range(inputs.size(1)):
        #    output, (hidden, cell) = self.step(inputs[:, i, :], hidden, cell)
        #return output.squeeze(), hidden, cell
        
        o, (hidden, cell) = self.lstm(inputs) #,(h_0, c_0))
        #print(o.shape)
        #output = self.out(o[-1,:,:].squeeze())
        
        return o, (hidden, cell)
    

#%% set weights and biases
torch.manual_seed(0)

batch_size = x_data.shape[1]
seq_len = x_data.shape[0]
n_inp = x_data.shape[2]
n_rec = lstm_weights[3].shape[1]//4
n_out = y_data.shape[1]

model = RNN(n_inp,n_rec,n_out)
model.lstm.weight_ih_l0 = nn.Parameter(torch.from_numpy(lstm_weights[3].T),requires_grad = False)
model.lstm.weight_hh_l0 = nn.Parameter(torch.from_numpy(lstm_weights[4].T),requires_grad = False)
model.lstm.bias_ih = nn.Parameter(torch.from_numpy(lstm_weights[6]),requires_grad = False)
model.lstm.bias_hh = nn.Parameter(torch.from_numpy(lstm_weights[5]),requires_grad = False)

#model.out.weight = nn.Parameter(torch.from_numpy(dense_weights[0]).double())
#model.out.bias = nn.Parameter(torch.from_numpy(dense_weights[1]).double())


#%% run model on data
model.eval()

with torch.no_grad():
    output, (h_n, c_n) = model(x_data)

# visualise hidden layer
# plt.figure()
# for i in range(output.shape[0]):
#     plt.subplot(3,6,i+1)
#     plt.imshow(output[i,:,:].detach().numpy())
#     plt.colorbar()
#%% PCA

trial_data = output
# time x batch x unit

# average across all unattended stimuli
trial_reds_up = trial_data[:,0:4,:].mean(axis=1)
trial_blues_up = trial_data[:,4:8,:].mean(axis=1)
trial_greens_up = trial_data[:,8:12,:].mean(axis=1)
trial_yellows_up = trial_data[:,12:16,:].mean(axis=1)

trial_reds_down = trial_data[:,16:20,:].mean(axis=1)
trial_blues_down = trial_data[:,20:24,:].mean(axis=1)
trial_greens_down = trial_data[:,24:28,:].mean(axis=1)
trial_yellows_down = trial_data[:,28:32,:].mean(axis=1)


trial_data = torch.stack((trial_reds_up, trial_blues_up, trial_greens_up, trial_yellows_up,
                        trial_reds_down, trial_blues_down, trial_greens_down, trial_yellows_down), axis=1)

delay1_pca = PCA(n_components=3) # Initializes PCA
delay2_pca = PCA(n_components=3) # Initializes PCA

# extract pre- and post-cue activity
delay1 = trial_data[9].detach()
delay1 -= torch.mean(delay1) # demean
delay2 = trial_data[17].detach()
delay2 -= torch.mean(delay2) # demean

# run PCA
delay1_3dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
delay2_3dcoords = delay2_pca.fit_transform(delay2)


# %% plotting pca - data in reduced dim space

plot_colours = ['b','g','y','r']

plt.figure()

ax = plt.subplot(121, projection='3d')
plot_geometry(ax, delay1_3dcoords, delay1_pca, plot_colours)
plt.title('pre-cue')


ax2 = plt.subplot(122, projection='3d')
plot_geometry(ax2, delay2_3dcoords, delay2_pca, plot_colours)
plt.title('post-cue')


equal_axes = True

if equal_axes:
    # equal x, y and z axis scale
    ax_lims = np.array(ax.xy_viewLim)
    ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))
    
    ax2_lims = np.array(ax2.xy_viewLim)
    ax2.set_xlim3d(np.min(ax2_lims),np.max(ax2_lims))
    ax2.set_ylim3d(np.min(ax2_lims),np.max(ax2_lims))
    ax2.set_zlim3d(np.min(ax2_lims),np.max(ax2_lims))

#%% add planes of best fit

n_colours = 4

#pre-cue
delay1_up_pca = PCA(n_components=2) # do PCA to get best-fit plane for delay 1 up (loc 1)
delay1_down_pca = PCA(n_components=2)

delay1_planeUp = delay1_up_pca.fit(delay1_3dcoords[0:n_colours,:]-np.mean(delay1_3dcoords[0:n_colours,:])) # get directions of max variance, i.e. vectors defining the plane
delay1_planeDown = delay1_down_pca.fit(delay1_3dcoords[n_colours:,:] - np.mean(delay1_3dcoords[n_colours:,:]))

# calculate angle between planes
cos_theta_pre = np.dot(np.cross(delay1_planeUp.components_[0,:],delay1_planeUp.components_[1,:]),
                   np.cross(delay1_planeDown.components_[0,:],delay1_planeDown.components_[1,:]))
print('Angle pre-cue: %.2f' %(np.degrees(np.arccos(cos_theta_pre))))
      
# plot_subspace(ax,delay1_3dcoords[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
# plot_subspace(ax,delay1_3dcoords[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)

# find the scaling factor for the planes - so that the plot looks nice
sorted_verts1, sorting_order = vops.sortByPathLength(delay1_3dcoords[:n_colours,:])
sorted_verts2, sorting_order = vops.sortByPathLength(delay1_3dcoords[n_colours:,:])

long_diag1 = np.linalg.norm(sorted_verts1[0]-sorted_verts1[2])
long_diag2 = np.linalg.norm(sorted_verts2[0]-sorted_verts2[2])
scale = 0.1*np.max([long_diag1,long_diag2])


plot_plane_old(ax,delay1_planeUp,delay1_3dcoords[:n_colours,:],scale,fc='k',a=0.2)
plot_plane_old(ax,delay1_planeDown,delay1_3dcoords[n_colours:,:],scale,fc='k',a=0.2)

#post-cue
delay2_up_pca = PCA(n_components=2)
delay2_down_pca = PCA(n_components=2)

delay2_planeUp = delay2_up_pca.fit(delay2_3dcoords[0:n_colours,:] - np.mean(delay2_3dcoords[0:n_colours,:]))
delay2_planeDown = delay2_down_pca.fit(delay2_3dcoords[n_colours:,:] - np.mean(delay2_3dcoords[n_colours:,:]))

cos_theta_post = np.dot(np.cross(delay2_planeUp.components_[0,:],delay2_planeUp.components_[1,:]),
                   np.cross(delay2_planeDown.components_[0,:],delay2_planeDown.components_[1,:]))
print('Angle post-cue: %.2f' %(np.degrees(np.arccos(cos_theta_post))))


#plot_subspace(ax2,delay2_3dcoords[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
#plot_subspace(ax2,delay2_3dcoords[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)

sorted_verts1, sorting_order = vops.sortByPathLength(delay2_3dcoords[:n_colours,:])
sorted_verts2, sorting_order = vops.sortByPathLength(delay2_3dcoords[n_colours:,:])

long_diag1 = np.linalg.norm(sorted_verts1[0]-sorted_verts1[2])
long_diag2 = np.linalg.norm(sorted_verts2[0]-sorted_verts2[2])
scale = 0.05*np.max([long_diag1,long_diag2])

plot_plane_old(ax2,delay2_planeUp,delay2_3dcoords[:n_colours,:],scale,fc='k',a=0.2)



plot_plane_old(ax2,delay2_planeDown,delay2_3dcoords[n_colours:,:],scale,fc='k',a=0.2)



# %% visualise xdata

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,sharex = True)
ax1.imshow(x_data[0,:,:])
ax1.set_xlabel('channel')
ax1.set_ylabel('trial')
ax1.set_title('Stim')

ax2.imshow(x_data[9,:,:])
ax2.set_title('Delay1, t=10')

ax3.imshow(x_data[10,:,:])
ax3.set_title('Retrocue, t=11')

ax4.imshow(x_data[-1,:,:])
ax4.set_title('Delay2, t=18')

#%% visualise all timesteps

plt.figure()
for t in range(x_data.shape[0]):
    plt.subplot(3,6,t+1)
    plt.imshow(x_data[t,:5,:]) # only plotting first 5 trials from batch


