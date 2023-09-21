#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:46:53 2021

@author: emilia
"""

rdm1 = np.mean(rdm_postcue_binned[:,:,5:7],-1)

rdm2 = np.mean(rdm_postcue_binned[:,:,0:2],-1)
rdm3 = np.mean(rdm_postcue_binned[:,:,[0,1,2,3,4,7,8,9]],-1)


#%%

plt.figure()
plt.subplot(121)
plt.imshow(rdm1)
plt.colorbar()
plt.title('180 degrees')

plt.subplot(122)
plt.imshow(rdm2)
plt.colorbar()
plt.title('0 degrees')


#%% mds for the above


mds1 = MDS(n_components=3, 
              metric=False, 
              dissimilarity='precomputed', 
              max_iter=1000)
mds2 = MDS(n_components=3, 
              metric=False, 
              dissimilarity='precomputed', 
              max_iter=1000)


mds_180 = mds1.fit_transform(rdm1)
mds_0 = mds2.fit_transform(rdm2)


#%%

mds_180 = fit_mds_to_rdm(rdm1)
mds_0 = fit_mds_to_rdm(rdm2)


#%%

colours = ['r','y','g','b']
n_colours = len(colours)

plt.figure()
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122, projection='3d')
plot_geometry(ax1,mds_180,[],colours)
plot_geometry(ax2,mds_0,[],colours)    

#%%%

# get directions of max variance, i.e. vectors defining the plane
mds180_planeUp = get_best_fit_plane(mds_180[0:n_colours])
mds180_planeDown = get_best_fit_plane(mds_180[n_colours:])

# calculate angle between planes
theta_pre = get_angle_between_planes(mds180_planeUp.components_,mds180_planeDown.components_)
print('Angle pre-cue: %.2f' %theta_pre)

plot_subspace(ax1,mds_180[:n_colours,:],mds180_planeUp.components_,fc='k',a=0.2)
plot_subspace(ax1,mds_180[n_colours:,:],mds180_planeDown.components_,fc='k',a=0.2)



mds0_planeUp = get_best_fit_plane(mds_0[0:n_colours])
mds0_planeDown = get_best_fit_plane(mds_0[n_colours:])

theta_pre = get_angle_between_planes(mds0_planeUp.components_,mds0_planeDown.components_)
print('Angle pre-cue: %.2f' %theta_pre)


plot_subspace(ax2,mds_0[:n_colours,:],mds0_planeUp.components_,fc='k',a=0.2)
plot_subspace(ax2,mds_0[n_colours:,:],mds0_planeDown.components_,fc='k',a=0.2)
#%%


u,s,v= np.linalg.svd(rdm1)


reconstruct = u[:,:3]@(np.diag(s[:3])@v[:3,:])


plt.figure()
plt.subplot(121)
plt.imshow(rdm1)
plt.colorbar()
plt.subplot(122)
plt.imshow(reconstruct)
plt.colorbar()


#%%

u,s,v= np.linalg.svd(rdm3)


reconstruct = u[:,:3]@(np.diag(s[:3])@v[:3,:])


plt.figure()
plt.subplot(121)
plt.imshow(rdm3)
plt.colorbar()
plt.subplot(122)
plt.imshow(reconstruct)
plt.colorbar()

#%%

mds_180 = fit_mds_to_rdm(rdm1)
mds_0 = fit_mds_to_rdm(rdm3)

#%%  plot mds
colours = ['r','y','g','b']
n_colours = len(colours)

plt.figure()
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122, projection='3d')
plot_geometry(ax1,mds_180,[],colours)
plot_geometry(ax2,mds_0,[],colours)    

#%% get planes


plane1 = get_best_fit_plane(mds_180[:n_colours])
plane2 = get_best_fit_plane(mds_180[n_colours:])
angle1 = get_angle_between_planes(plane1.components_,plane2.components_)

plane3 = get_best_fit_plane(mds_0[:n_colours])
plane4 = get_best_fit_plane(mds_0[n_colours:])
angle2 = get_angle_between_planes(plane3.components_,plane4.components_)

#%%
u,s,v= np.linalg.svd(rdm1)
reconstruct = u[:,:3]@(np.diag(s[:3])@v[:3,:])

mds_180_3 = fit_mds_to_rdm(reconstruct)


u,s,v= np.linalg.svd(rdm3)
reconstruct = u[:,:3]@(np.diag(s[:3])@v[:3,:])

mds_0_3 = fit_mds_to_rdm(reconstruct)


#%% rectangles with matched longer sides


# models


post_up = np.array([[1.5,1.5,-1,-1],[1,1,1,1],[1,-1,-1,1]]).T
post_down = np.array([[1.5,1.5,-1,-1],[-1,-1,-1,-1],[1,-1,-1,1]]).T
post = np.concatenate((post_up,post_down),axis=0)



plt.figure()
ax = plt.subplot(111, projection='3d')

cols = ['r','y','g','b']

plot_geometry(ax, post, [], cols)
    
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])


up1 = np.stack((np.zeros((3,)),plane1.components_[0,:]))
up2 = np.stack((np.zeros((3,)),plane1.components_[1,:]))
down1 = np.stack((np.zeros((3,)),plane2.components_[0,:]))
down2 = np.stack((np.zeros((3,)),plane2.components_[1,:]))

angle = get_angle_between_planes(plane1.components_,plane2.components_)

ax.plot(up1[:,0],up1[:,1],up1[:,2],'r-')
ax.plot(up2[:,0],up2[:,1],up2[:,2],'b-')
ax.plot(down1[:,0],down1[:,1],down1[:,2],'r--')
ax.plot(down2[:,0],down2[:,1],down2[:,2],'b--')



#%% rectangles with mismatched longer sides

post_up = np.array([[1.5,1.5,-1,-1],[1,1,1,1],[1,-1,-1,1]]).T
post_down = np.array([[1,1,-1,-1],[-1,-1.5,-1.5,-1],[1,-1,-1,1]]).T
post = np.concatenate((post_up,post_down),axis=0)



plt.figure()
ax = plt.subplot(111, projection='3d')

cols = ['r','y','g','b']

plot_geometry(ax, post, [], cols)
    
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])


up1 = np.stack((np.zeros((3,)),plane1.components_[0,:]))
up2 = np.stack((np.zeros((3,)),plane1.components_[1,:]))
down1 = np.stack((np.zeros((3,)),plane2.components_[0,:]))
down2 = np.stack((np.zeros((3,)),plane2.components_[1,:]))

angle = get_angle_between_planes(plane1.components_,plane2.components_)

ax.plot(up1[:,0],up1[:,1],up1[:,2],'r-')
ax.plot(up2[:,0],up2[:,1],up2[:,2],'b-')
ax.plot(down1[:,0],down1[:,1],down1[:,2],'r--')
ax.plot(down2[:,0],down2[:,1],down2[:,2],'b--')

print(angle)


plot_subspace(ax,post_up,plane1.components_,fc='k',a=0.2)
plot_subspace(ax,post_down,plane2.components_,fc='k',a=0.2)



# ax.plot(vops.makeVec(n1)[:,0],vops.makeVec(n1)[:,1],vops.makeVec(n1)[:,2],'g--')
# ax.plot(vops.makeVec(n2)[:,0],vops.makeVec(n2)[:,1],vops.makeVec(n2)[:,2],'y--')


# ax.plot(vops.makeVec(new_norm2)[:,0],vops.makeVec(new_norm2)[:,1],vops.makeVec(new_norm2)[:,2],'k--')
#%%

# plane1_aligned,plane2_aligned,n  = align_plane_vecs(post_up, post_down, plane1, plane2)


plane1_aligned,plane2_aligned,n  = align_plane_vecs(delay2_3dcoords[:n_colours,:],
                                                    delay2_3dcoords[n_colours:,:],
                                                    delay2_planeUp,
                                                    delay2_planeDown)






#%%

ap,n = align_plane_vecs(delay2_3dcoords[:n_colours,:], 
                      delay2_planeUp)



#%%

ax.scatter(points2_proj[0,0],points2_proj[0,1],points2_proj[0,2],marker='o',c='k')    
ax.scatter(points2_proj[1,0],points2_proj[1,1],points2_proj[1,2],marker='s',c='k')    



d11 = np.linalg.norm(plane1.components_[0,:] - v11)
d12 = np.linalg.norm(plane1.components_[1,:] - v11)

d21 = np.linalg.norm(plane2.components_[0,:] - v21)
d22 = np.linalg.norm(plane2.components_[1,:] - v21)


v21 = getVecFromPoints(points2_proj[0,:],points2_proj[1,:])
v22 = getVecFromPoints(points2_proj[0,:],points2_proj[3,:])
v21 /= np.linalg.norm(v21)
v22 /= np.linalg.norm(v22)

n2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])
n1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
new_basis = np.concatenate((plane2.components_,np.expand_dims(n2,0)),axis=0)
v21_newbasis = new_basis @ v21


ax.plot(vops.makeVec(v21)[:,0],vops.makeVec(v21)[:,1],vops.makeVec(v21)[:,2],'k-->')

ax.plot(vops.makeVec(v22)[:,0],vops.makeVec(v22)[:,1],vops.makeVec(v22)[:,2],'g-->')

#%%

post_up = np.array([[1.5,1.5,-1,-1],[1,1,1,1],[1,-1,-1,1]]).T
post_down = np.array([[1,1,-1,-1],[-1,-1,-1,-1],[1,-1.5,-1.5,1]]).T
post = np.concatenate((post_up,post_down),axis=0)



plt.figure()
ax = plt.subplot(111, projection='3d')

cols = ['r','y','g','b']

plot_geometry(ax, post, [], cols)
    
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])


up1 = np.stack((np.zeros((3,)),plane1.components_[0,:]))
up2 = np.stack((np.zeros((3,)),plane1.components_[1,:]))
down1 = np.stack((np.zeros((3,)),plane2.components_[0,:]))
down2 = np.stack((np.zeros((3,)),plane2.components_[1,:]))

norm1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
norm2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])

angle = get_angle_between_planes(plane1.components_,plane2.components_)

ax.plot(up1[:,0],up1[:,1],up1[:,2],'r-',label='x1')
ax.plot(up2[:,0],up2[:,1],up2[:,2],'b-',label='y1')
ax.plot(down1[:,0],down1[:,1],down1[:,2],'r--',label='x2')
ax.plot(down2[:,0],down2[:,1],down2[:,2],'b--',label='y2')

print(angle)
plt.legend(bbox_to_anchor=[1.5,1])


#%%

#%%

post_up = np.array([[1.5,1.5,-1,-1],[1,1,1,1],[1,-1,-1,1]]).T
post_down = np.array([[1,1,-1,-1],[-1,-1,-1,-1],[1,-1.5,-1.5,1]]).T
post = np.concatenate((post_up,post_down),axis=0)



plt.figure()
ax = plt.subplot(111, projection='3d')

cols = ['r','y','g','b']

plot_geometry(ax, post, [], cols)
    
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])


up1 = np.stack((np.zeros((3,)),plane1.components_[0,:]))
up2 = np.stack((np.zeros((3,)),plane1.components_[1,:]))
down1 = np.stack((np.zeros((3,)),plane2.components_[0,:]))
down2 = np.stack((np.zeros((3,)),plane2.components_[1,:]))

norm1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
norm2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])

angle = get_angle_between_planes(plane1.components_,plane2.components_)

ax.plot(up1[:,0],up1[:,1],up1[:,2],'r-',label='x1')
ax.plot(up2[:,0],up2[:,1],up2[:,2],'b-',label='y1')
ax.plot(down1[:,0],down1[:,1],down1[:,2],'r--',label='x2')
ax.plot(down2[:,0],down2[:,1],down2[:,2],'b--',label='y2')

print(angle)
plt.legend(bbox_to_anchor=[1.5,1])

#%%%%






#%%%%%
import numpy as np

#%%


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





 
# calculate and save the RDM
flipped_RDM = squareform(pdist(post_flipped))




v1 = vops.makeVec(plane1.components_[0])
v2 = vops.makeVec(plane1.components_[1])

ax.plot(v1[:,0],v1[:,1],v1[:,2],'r-',label='v1')
ax.plot(v2[:,0],v2[:,1],v2[:,2],'b-',label='v2')



v3 = vops.makeVec(plane2.components_[0])
v4 = vops.makeVec(plane2.components_[1])

ax.plot(v3[:,0],v3[:,1],v3[:,2],'k--',label='v3')
ax.plot(v4[:,0],v4[:,1],v4[:,2],'y--',label='v4')

#%%


post_flipped[:2,-1] += 0.1

post_flipped[2:4,-1] -= 0.1


post_flipped[4:6,-1] -= 0.1

post_flipped[6:,-1] += 0.1

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post_flipped,colours,legend_on=False)
# ax.set_title('mirror')


ax.scatter(0,0,0,marker='+',c='k')



plane1 = get_best_fit_plane(post_flipped[0:n_colours])
plane2 = get_best_fit_plane(post_flipped[n_colours:]) 

angle = get_angle_between_planes(plane1.components_,plane2.components_)
print('Angle: %.2f' %angle)


#%%

# post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
# post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

# post = np.concatenate((post_up,post_down),axis=1).T



# post[:2,-1] += 0.1

# post[2:4,-1] -= 0.1


# post[4:6,-1] -= 0.1

# post[6:,-1] += 0.1


plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post,colours,legend_on=False)

plane1 = get_best_fit_plane(post[0:n_colours])
plane2 = get_best_fit_plane(post[n_colours:]) 

angle = get_angle_between_planes(plane1.components_,plane2.components_)
print('Angle: %.2f' %angle)



ax.set_title('phase-aligned ['+str(np.round(angle))+'°]')


#%%

post = np.concatenate((post_up,post_down),axis=1).T

post[0,0:2] += .1 
post[2,0:2] -= .1 

post[5,:2] += .1
post[7,:2] -= .1

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post,colours,legend_on=False)

plane1 = get_best_fit_plane(post[0:n_colours])
plane2 = get_best_fit_plane(post[n_colours:]) 

angle = get_angle_between_planes(plane1.components_,plane2.components_)
print('Angle: %.2f' %angle)



# ax.set_title('phase-aligned ['+str(np.round(angle))+'°]')


# v1 = vops.makeVec(plane1.components_[0])
# v2 = vops.makeVec(plane1.components_[1])

# ax.plot(v1[:,0],v1[:,1],v1[:,2],'r-',label='v1')
# ax.plot(v2[:,0],v2[:,1],v2[:,2],'b-',label='v2')



# v3 = vops.makeVec(plane2.components_[0])
# v4 = vops.makeVec(plane2.components_[1])

# ax.plot(v3[:,0],v3[:,1],v3[:,2],'k--',label='v3')
# ax.plot(v4[:,0],v4[:,1],v4[:,2],'y--',label='v4')

# plt.legend()


#%
#%%
ap,n = align_plane_vecs(post[:n_colours,:], 
                      plane1)
ap2,n2 = align_plane_vecs(post[n_colours:,:], 
                      plane2)

angle = get_angle_between_planes(ap,ap2)
print(angle)




ax.set_title('phase-aligned corrected ['+str(np.round(angle))+'°]')


v1 = vops.makeVec(ap[0])
v2 = vops.makeVec(ap[1])

ax.plot(v1[:,0],v1[:,1],v1[:,2],'r-',label='av1')
ax.plot(v2[:,0],v2[:,1],v2[:,2],'b-',label='av2')



v3 = vops.makeVec(ap2[0])
v4 = vops.makeVec(ap2[1])

ax.plot(v3[:,0],v3[:,1],v3[:,2],'k--',label='av3')
ax.plot(v4[:,0],v4[:,1],v4[:,2],'y--',label='av4')

plt.legend()


#%%

ap,n = align_plane_vecs(post_flipped[:n_colours,:], 
                      plane1)
ap2,n2 = align_plane_vecs(post_flipped[n_colours:,:], 
                      plane2)

angle = get_angle_between_planes(ap,ap2)
print(angle)

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,post_flipped,colours,legend_on=False)
# ax.set_title('mirror')


ax.scatter(0,0,0,marker='+',c='k')

# plane1 = get_best_fit_plane(post_flipped[0:n_colours])
# plane2 = get_best_fit_plane(post_flipped[n_colours:]) 

# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Angle: %.2f' %angle)


ax.set_title('flipped corrected ['+str(np.round(angle))+'°]')


v1 = vops.makeVec(ap[0])
v2 = vops.makeVec(ap[1])

ax.plot(v1[:,0],v1[:,1],v1[:,2],'r-',label='av1')
ax.plot(v2[:,0],v2[:,1],v2[:,2],'b-',label='av2')



v3 = vops.makeVec(ap2[0])
v4 = vops.makeVec(ap2[1])

ax.plot(v3[:,0],v3[:,1],v3[:,2],'k--',label='av3')
ax.plot(v4[:,0],v4[:,1],v4[:,2],'y--',label='av4')

plt.legend()



v5 = vops.makeVec(n)
v6 =vops.makeVec(n2)


ax.plot(v5[:,0],v5[:,1],v5[:,2],'g-',label='n1')
ax.plot(v6[:,0],v6[:,1],v6[:,2],'g--',label='n2')

plt.legend()


#%%

def align_plane_vecs(points, pca):
    
    
    #x = pca.components_[0,:] # PC1
    y = pca.components_[1,:] # PC2
    
    # project points onto the plane
    points_proj = np.zeros(points.shape)   
    com = np.mean(points, axis=0) # centre of mass
    for i in range(points_proj.shape[0]):
        points_proj[i,:] = vops.getProjection(points[i,:]-com,pca.components_)
    
    
    # get the vector corresponding to the side of the parallelogram
    a = vops.getVecFromPoints(points_proj[0,:],points_proj[1,:])
    
    # normalise it
    a /= np.linalg.norm(a)
    
    # do change of basis to PCA-given axes
    plane_basis = np.concatenate((pca.components_,
                                  np.cross(pca.components_[0,:],pca.components_[1:,])))
    
    a_newBasis = plane_basis @ a
    
    
    # # calculate the angle of rotation
    # theta = np.arccos(a_newBasis[0])
    
    # # calculate the angle between the new axis 'a' and original 'y'
    # alpha =  np.arcsin(a[0])
    # if alpha < 0:
    #     # flip the y axis
    #     y = -y
    
    # # theta = np.pi/2
    
    R = np.array([[0,1,0],[-1,0,0],[0,0,1]]).T
    b_newBasis = R @ a_newBasis             # rotate a by 90 degrees to get b
    # # # rotate the y-axis by theta
    # b_newBasis = np.array([-np.sin(theta),np.cos(theta),0])
    
    # return to standard basis
    b = plane_basis.T @ b_newBasis
        
    #check that the new vectors are orhogonal
    if np.dot(a,b) > 0.001:
        # print(y)
        # print(np.degrees(theta))
        print(np.degrees(np.dot(a,b)))
        raise ValueError('New vectors not orthogonal')
    # calculate the cross-product
    n = np.cross(a,b)
    
    plane_vecs_aligned = np.stack((a,b))
        
    return plane_vecs_aligned,n