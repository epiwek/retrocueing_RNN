#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:01:58 2021

@author: emilia
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import rep_geom
import helpers

import pdb

#%%

def plot_settings():
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

#%% experimental set up

def plot_example_stimulus(data,**kwargs):
    # plt.figure()
    
    # if len(data.shape)<3:
    #     # plot a specific trial
    #     plt.imshow(data,*kwargs)
    # else:
    #     # pick a random trial to plot
    #     ix = np.random.randint(data.shape[1])
    #     plt.imshow(data[:,ix,:],**kwargs)
    # plt.colorbar
    
    fig, ax = plt.subplots()
    
    if len(data.shape)<3:
        # plot a specific trial
        cax = ax.imshow(data,**kwargs)
    else:
        # pick a random trial to plot
        ix = np.random.randint(data.shape[1])
        cax = ax.imshow(data[:,ix,:],**kwargs)
    fig.colorbar(cax,ticks=[0,1])
    return fig,ax, cax

#%% paired data

def add_jitter(data,jitter='auto'):
    # check that data is a vector
    if len(data.shape)>1:
        raise ValueError('Input data should be 1D')
    
    unique = np.unique(data)
    x_jitter = np.zeros(data.shape)
    # check for duplicate vals
    if len(unique) != len(data) :
        # there are duplicates
        for i in range(len(unique)):
            # save indices of duplicates
            dups_ix = np.where(data==unique[i])[0]
            n_dups = len(dups_ix)
            if jitter=='auto':
                x_jitter[dups_ix] = np.linspace(-.2,.2,n_dups)
            else:
                # custom jitter value
                x_jitter[dups_ix] = \
                    np.arange(0,n_dups*jitter,jitter) \
                    - np.mean(np.arange(0,n_dups*jitter,jitter))
    return x_jitter


def plot_paired_data(x_data,y_data,ax,colours,jitter='auto',**kwargs):
    # determine x jitter
    x_jitter = np.zeros(y_data.shape)
    if len(y_data.shape) == 1:
        x_jitter = np.expand_dims(x_jitter,0)
        y_data = np.expand_dims(y_data,0)
        x_data = np.expand_dims(x_data,0)
    
    for i in range(2):
        x_jitter[:,i] = add_jitter(y_data[:,i],jitter=jitter)
    
    #plot
    # loop over samples (models)
    for i in range(y_data.shape[0]):
        ax.plot(x_data[i,:]+x_jitter[i,:],y_data[i,:],'k-',**kwargs)
        # loop over the pair
        for j in range(y_data.shape[1]):
            ax.plot(x_data[i,j]+x_jitter[i,j],y_data[i,j],'o',
                    color = colours[j],**kwargs)
            
#%% 3D data and planes
def plot_geometry(ax,Z,pca,plot_colours,plot_outline = True,legend_on = True, **kwargs):
    """
    Plot 3D data for one or two subspaces.
    
    Parameters
    ----------
    
    ax :  

        

    """
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(Z[:n_colours,0],Z[0,0]),
              np.append(Z[:n_colours,1],Z[0,1]),
              np.append(Z[:n_colours,2],Z[0,2]),'k-',**kwargs)
    ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
              c='k',label='loc1',**kwargs)
    ax.scatter(Z[:n_colours,0],Z[:n_colours,1],
              Z[:n_colours,2],marker='o',s = 40,c=plot_colours,**kwargs)
  
    # repeat for loc 2 - if data supplied
    if Z.shape[0]>n_colours:
        if plot_outline:
            ax.plot(np.append(Z[n_colours:,0],Z[n_colours,0]),
                  np.append(Z[n_colours:,1],Z[n_colours,1]),
                  np.append(Z[n_colours:,2],Z[n_colours,2]),'k-',**kwargs)
        ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
                  c='k',label='loc2',**kwargs)
        ax.scatter(Z[n_colours:,0],Z[n_colours:,1],
                  Z[n_colours:,2],marker='s',s = 40,c=plot_colours,**kwargs)
        
    # add the PVEs for each PC to the corresponding axis - if data supplied
    if pca:
        ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
        ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
        ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')
    
    if legend_on:
        ax.legend()


def plot_plane(ax,verts,fc='k',a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
    
def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check the shape of the data matrix - should be (n_points,3)')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.defPlaneShape(verts,plane_vecs)
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
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
    
#%% other plots
def shadow_plot(ax,x,y, precalc = False, alpha = 0.3, **kwargs):
    
    if precalc:
        y_mean = y[0]
        y_sem = y[1]
    else:
        y_mean = np.mean(y,axis=0)
        y_sem = np.std(y,axis=0)/np.sqrt(len(y))
    # pdb.set_trace()
    H1 = ax.fill_between(x,y_mean+y_sem,y_mean-y_sem,
                    alpha = alpha, **kwargs)
    H2 = ax.plot(x,y_mean,**kwargs)
    return H1,H2

    