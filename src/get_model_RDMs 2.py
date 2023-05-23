#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:18:33 2021

@author: emilia
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import pickle
import os.path
import vec_operations as vops 
from rep_geom import *
import constants
from helpers import check_path



path = constants.PARAMS['FULL_PATH']
#%% define plotting functions

def plot_geometry(ax,points,plot_colours,plot_outline = True,legend_on=True):
    ms = 75
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(points[:n_colours,0],points[0,0]),
              np.append(points[:n_colours,1],points[0,1]),
              np.append(points[:n_colours,2],points[0,2]),'k-')
    ax.scatter(points[0,0],points[0,1], points[0,2],marker='^',s = ms,
              c='k',label='loc1')
    ax.scatter(points[:n_colours,0],points[:n_colours,1],
              points[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(points[n_colours:,0],points[n_colours,0]),
              np.append(points[n_colours:,1],points[n_colours,1]),
              np.append(points[n_colours:,2],points[n_colours,2]),'k-')
    ax.scatter(points[-1,0],points[-1,1], points[-1,2],marker='s',s = ms,
              c='k',label='loc2')
    ax.scatter(points[n_colours:,0],points[n_colours:,1],
              points[n_colours:,2],marker='s',s = ms,c=plot_colours)
    if legend_on:
        ax.legend()

def plot_plane(ax,verts,fc='k',a=0.2):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts

#%% pick coordinates
import seaborn as sns


# colours = ['r','y','g','b']
colours = sns.color_palette("husl",4)
n_colours=len(colours)


# pre_up = np.array([[1,1,-1,-1],[-1,1,1,-1],[0,0,0,0]])
# pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])

pre_up = np.array([[-1,-1,1,1],[1,-1,-1,1],[0,0,0,0]])
pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])

pre = np.concatenate((pre_up,pre_down),axis=1)

# post - both rotate

post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

post = np.concatenate((post_up,post_down),axis=1)



# no post-cue plane distance RDM
post_singlePlane = np.concatenate([post[:,:n_colours].T]*2) # coordinates

#%% plot ortho and parallel

plt.figure(figsize=(12,5))

ax = plt.subplot(121, projection='3d')
plot_geometry(ax,pre.T,colours,legend_on=False)
ax.set_title('pre-cue')
ax.scatter(0,0,0,marker='+',c='k')


ax2 = plt.subplot(122, projection='3d')
plot_geometry(ax2,post.T,colours)
ax.set_title('pre-cue')
ax2.set_title('post-cue')
plt.legend(bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure)

ax2.scatter(0,0,0,marker='+',c='k')

fig_path = path+'RSA'
check_path(fig_path)

plt.savefig(fig_path+'/geom_predictions_from_paper')


#%% plot planes - to explain how angle is calculated

colours = ['r','y','g','b']
plt.figure(figsize=(5,5))

ax = plt.subplot(111, projection='3d')
plot_geometry(ax,pre.T,colours,legend_on=False)
ax.set_title('pre-cue')
ax.scatter(0,0,0,marker='+',c='k')

plot_plane(ax,pre_up.T,fc='k',a=0.2)
plot_plane(ax,pre_down.T,fc='k',a=0.2)


ax.plot([0,0],[0,0],[0,1],'r-',label='normal1')
ax.plot([0,1],[0,0],[0,0],'r--',label='normal2')

plt.xticks([-1,0,1])
plt.yticks([-1,0,1])

ax.set_zticks([-1,0,1])

plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
#%% get model RDMs

orthoPlanes_RDM = squareform(pdist(pre.T,))#metric='correlation'))
parallelPlanes_RDM = squareform(pdist(post.T))#,metric='correlation'))
singlePlane_RDM = squareform(pdist(post_singlePlane))#,metric='correlation'))


model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM,singlePlane_RDM),axis=2)
# model_RDMs /= np.max(model_RDMs) # normalise values to be within 0 and 1



#%% double-check that they correspond to the right geometry


from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist


def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)


mds_ortho = fit_mds_to_rdm(orthoPlanes_RDM)
mds_parallel = fit_mds_to_rdm(parallelPlanes_RDM)
mds_single= fit_mds_to_rdm(singlePlane_RDM)


#%% plot model RDMs

fig, axes = plt.subplots(1,3, sharex=True, sharey = True, figsize=(10,50))


titles = ['ortho','parallel','single']
for ix,ax in enumerate(axes.flat):
    im = ax.imshow(model_RDMs[:,:,ix],
                   vmin=np.min(model_RDMs),
                   vmax=np.max(model_RDMs),
                   # cmap = sns.color_palette("flare_r",as_cmap = True))
                   cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    ax.set_title(titles[ix])
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im,ax=axes.ravel().tolist())


#%% save model RDMs

# get upper diagonal / square form
model_RDMs_sqform = np.zeros((squareform(model_RDMs[:,:,0]).shape[0],model_RDMs.shape[-1]))
for model in range(model_RDMs.shape[-1]):
    model_RDMs_sqform[:,model] = squareform(model_RDMs[:,:,model])


if not (os.path.exists(path+'RSA')):
            os.mkdir(path+'RSA')

save_path = path + 'RSA/'


pickle.dump(model_RDMs,open(save_path + 'model_RDMs.pckl','wb'))
pickle.dump(model_RDMs_sqform,open(save_path + 'model_RDMs_sqform.pckl','wb'))

model_RDMs_order = ['orthogonal','parallel','single']

pickle.dump(model_RDMs_order,open(save_path + 'model_RDMs_order.pckl','wb'))


#%%
# # ROTATIONS

# take subset of points
# points = post.T[n_colours:,:]
# theta_degrees=45
def rotatePlaneByAngle(points,theta_degrees):
    theta = np.radians(theta_degrees)

    # want to rotate around the y axis

    # construct rotation matrix
    R = np.eye(3)
    # z component stays the same
    R[:,0] = np.array([np.cos(theta),np.sin(theta),0])
    R[:,1] = np.array([-np.sin(theta),np.cos(theta),0])

    # apply rotation to points

    points_rotated = R @ points.T
    
    return points_rotated

# points_r = rotatePlaneByAngle(points,theta_degrees)
# new_points = np.concatenate((post[:,:n_colours],points_r),axis=1)

# plt.figure()
# ax = plt.subplot(111,projection='3d')
# plot_geometry(ax,new_points.T,colours)


# create a bunch of rotation matrices, delta_theta = 5 degrees
save_path = path + 'RSA/'

if not (os.path.exists(save_path+'rotated_coords/')):
    os.mkdir(save_path+'rotated_coords/')
if not (os.path.exists(save_path+'rotated_fullRDMs/')):
    os.mkdir(save_path+'rotated_fullRDMs/')
if not (os.path.exists(save_path+'rotated_diagRDMs/')):
    os.mkdir(save_path+'rotated_diagRDMs/')
    
if not (os.path.exists(save_path+'ortho_rotated_coords/')):
    os.mkdir(save_path+'ortho_rotated_coords/')
if not (os.path.exists(save_path+'ortho_rotated_fullRDMs/')):
    os.mkdir(save_path+'ortho_rotated_fullRDMs/')
if not (os.path.exists(save_path+'ortho_rotated_diagRDMs/')):
    os.mkdir(save_path+'ortho_rotated_diagRDMs/')



theta_range = np.arange(0,365,5)

for i,theta_degrees in enumerate(theta_range):
    points = post.T[n_colours:,:]
    points_r = rotatePlaneByAngle(points,theta_degrees)
    
    new_coords = np.concatenate((post[:,:n_colours],points_r),axis=1)
    
    # save the coords
    pickle.dump(new_coords,open(save_path 
                                + 'rotated_coords/rotatedBy'
                                + str(theta_degrees)+'.pckl','wb'))
    
    # calculate and save the RDM
    rotated_RDM = squareform(pdist(new_coords.T))
    #full
    pickle.dump(rotated_RDM,open(save_path 
                                + 'rotated_fullRDMs/'
                                + str(theta_degrees)+'.pckl','wb'))
    # diagonal
    rotated_RDM_sqform = squareform(rotated_RDM)
    pickle.dump(rotated_RDM_sqform,open(save_path 
                            + 'rotated_diagRDMs/'
                            + str(theta_degrees)+'.pckl','wb'))
    
    ortho_points = pre.T[:n_colours,:]
    ortho_points_r = rotatePlaneByAngle(ortho_points,theta_degrees)
    new_ortho_coords = np.concatenate((ortho_points_r,pre[:,n_colours:]),axis=1)
    
    # save the coords
    pickle.dump(new_ortho_coords,open(save_path 
                                + 'ortho_rotated_coords/rotatedBy'
                                + str(theta_degrees)+'.pckl','wb'))
    
    # calculate and save the RDM
    ortho_rotated_RDM = squareform(pdist(new_ortho_coords.T))
    #full
    pickle.dump(ortho_rotated_RDM,open(save_path 
                                + 'ortho_rotated_fullRDMs/'
                                + str(theta_degrees)+'.pckl','wb'))
    # diagonal
    ortho_rotated_RDM_sqform = squareform(ortho_rotated_RDM)
    pickle.dump(ortho_rotated_RDM_sqform,open(save_path 
                            + 'ortho_rotated_diagRDMs/'
                            + str(theta_degrees)+'.pckl','wb'))


# save the theta range used

pickle.dump(theta_range,open(save_path+'theta_range.pckl','wb'))




#%% plot parallel variations - rotation by 90 degrees, mirror image and ortho 45
plt.rcParams.update({'font.size': 25})

plt.figure(figsize=(18,5))
ax = plt.subplot(131,projection='3d')

theta_degrees = 90
new_coords = pickle.load(open(save_path 
                                + 'rotated_coords/rotatedBy'
                                + str(theta_degrees)+'.pckl','rb'))


plot_geometry(ax,new_coords.T,colours,legend_on=False)
ax.set_title('$parallel_{90}$')

ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])
ax.set_zticks([-.5,0,.5])

ax.set_xlabel('dim 1',labelpad=20.0)
ax.set_ylabel('dim 2',labelpad=20.0)
ax.set_zlabel('dim 3',labelpad=30.0)

ax.tick_params(axis='z',which='major', pad=10)



R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
post_flipped = post.copy()
post_flipped = post_flipped.T

ax2 = plt.subplot(132,projection='3d')
plot_geometry(ax2,post_flipped,colours,legend_on=False)
ax2.set_title('mirror')

ax2.set_xlabel('dim 1',labelpad=25.0)
ax2.set_ylabel('dim 2',labelpad=25.0)
ax2.set_zlabel('dim 3',labelpad=30.0)


ax2.set_xticks([-1,0,1])
ax2.set_yticks([-1,0,1])
ax2.set_zticks([-.5,0,.5])

ax2.tick_params(which='major', pad=5)
ax2.tick_params(axis='z', pad=10)




ax3 = plt.subplot(133,projection='3d')
theta_degrees =45
new_coords = pickle.load(open(save_path 
                                + 'ortho_rotated_coords/rotatedBy'
                                + str(theta_degrees)+'.pckl','rb'))
plot_geometry(ax3,new_coords.T,colours)


ax3.set_title('$ortho_{45}$')
ax3.set_xlabel('dim 1',labelpad=20.0)
ax3.set_ylabel('dim 2',labelpad=20.0)
ax3.set_zlabel('dim 3',labelpad=20.0)


# plt.legend(bbox_to_anchor=(.8,.8))
plt.legend(bbox_to_anchor=(1.5,1.2))

plt.tight_layout()



#%%   MIRROR IMAGES


R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
post_flipped = post.copy()
post_flipped = post_flipped.T
post_flipped[n_colours:,:] = post_flipped[n_colours:,:]@R

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post_flipped,colours,legend_on=False)
# ax.set_title('mirror')


ax.scatter(0,0,0,marker='+',c='k')

plane1 = get_best_fit_plane(post_flipped[0:n_colours])
plane2 = get_best_fit_plane(post_flipped[n_colours:]) 

angle = get_angle_between_planes(plane1.components_,plane2.components_)
print('Angle: %.2f' %angle)


ax.set_title('flipped ['+str(np.round(angle))+'°]')
plt.savefig(fig_path+'/geom_flipped')

# save the coords
pickle.dump(post_flipped,open(save_path 
                             + 'flipped_coords'+'.pckl','wb'))



 
# calculate and save the RDM
flipped_RDM = squareform(pdist(post_flipped))
#full
pickle.dump(flipped_RDM,open(save_path 
                            + 'flipped_fullRDM.pckl','wb'))
# diagonal
flipped_RDM_sqform = squareform(flipped_RDM)
pickle.dump(flipped_RDM_sqform,open(save_path 
                        + 'flipped_RDM_sqform.pckl','wb'))


#%% post-cue phase-aligned

from rep_geom import *

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post.T,colours,legend_on = False)

ax.scatter(0,0,0,marker='+',c='k')

plane1 = get_best_fit_plane(post[:,0:n_colours].T)
plane2 = get_best_fit_plane(post[:,n_colours:].T) 

angle = get_angle_between_planes(plane1.components_,plane2.components_)

ax.set_title('phase-aligned ['+str(np.round(angle))+'°]')

plt.savefig(fig_path+'/geom_phaseAligned')


print('Angle: %.2f' %angle)
#%% check post-cue phase-misaligned - what would the plane angle be?
# mirror image







# #%% rotated 180


# R = np.array([[0,0,-1],[0,1,0],[1,0,0]])
# post_rotated180 = post.copy()
# post_rotated180 = post_rotated180.T
# post_rotated180[4:,:] = post_rotated180[4:,:]@R

# post_rotated180[4:,:] = post_rotated180[4:,:]@R

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# plot_geometry(ax,post_rotated180,colours,legend_on=False)
# # plt.legend(bbox_to_anchor=(1, 1),
# #             bbox_transform=plt.gcf().transFigure)

# ax.scatter(0,0,0,marker='+',c='k')

# plane1 = get_best_fit_plane(post_rotated180[0:n_colours])
# plane2 = get_best_fit_plane(post_rotated180[n_colours:]) 

# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Angle: %.2f' %angle)

# ax.set_title('rotated 180° ['+str(np.round(angle))+'°]')

# plt.savefig(fig_path+'/geom_rotated180')


# #%% rotated 90

# post_rotated90 = post.copy()
# post_rotated90 = post_rotated90.T
# post_rotated90[4:,:] = post_rotated90[[5,6,7,4],:]


# plt.figure()
# ax = plt.subplot(111, projection='3d')
# plot_geometry(ax,post_rotated90,colours,legend_on=False)
# # ax.set_title('rotated 90')


# ax.scatter(0,0,0,marker='+',c='k')

# plane1 = get_best_fit_plane(post_rotated90[0:n_colours])
# plane2 = get_best_fit_plane(post_rotated90[n_colours:]) 

# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Angle: %.2f' %angle)

# ax.set_title('rotated 90° ['+str(np.round(angle))+'°]')

# plt.savefig(fig_path+'/geom_rotated90')

# #%% # add more model RDMs



# # rotations
# post_rotated90_RDM = squareform(pdist(post_rotated90))
# post_rotated180_RDM = squareform(pdist(post_rotated180))
# post_flipped1_RDM = squareform(pdist(post_flipped1))
# post_flipped2_RDM = squareform(pdist(post_flipped2))


# model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM,post_singlePlane_RDM,
#                        post_rotated90_RDM,post_rotated180_RDM,
#                        post_flipped1_RDM,post_flipped2_RDM),axis=2)
# # # z-score - separately fir each model 
# # model_RDMs = (model_RDMs-np.mean(model_RDMs.reshape
# #                                  (model_RDMs.shape[0]*model_RDMs.shape[1],
# #                                   model_RDMs.shape[-1]),0)) / \
# #                                 np.std(model_RDMs.reshape(model_RDMs.shape[0]*
# #                                 model_RDMs.shape[1],model_RDMs.shape[-1]),0)

# # # check that it has worked:
    
# # if not np.all(np.round(np.mean(np.mean(model_RDMs,axis=0),axis=0)) == np.zeros(model_RDMs.shape[-1])):
# #     raise ValueError('There was a problem with z-scoring the model RDMs - mean not 0')

# # if  not np.all(np.round(np.std(np.std(model_RDMs,axis=0),axis=0)) == np.zeros(model_RDMs.shape[-1])):
# #     raise ValueError('There was a problem with z-scoring the model RDMs - std not 0')
# #     # should it be 0 and not 1?






# #%%


# # ,
# #                     'post_rotated90','post_rotated180',
# #                     'post_flipped1','post_flipped2']