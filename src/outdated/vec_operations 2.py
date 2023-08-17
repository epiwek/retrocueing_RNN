#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:07:28 2020

@author: emilia
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import ConvexHull
import pdb

def makeVec(point):
    dim = point.shape[0]
    vec = np.stack((np.zeros((dim,)),point))
    return vec

def makePoint(vec):
    point = vec[-1,:]
    return point

def getVecFromPoints(point1,point2):
    vec = point2-point1
    return vec
    
def getProjection(point,plane_vecs):
    '''
    

    Parameters
    ----------
    point : TYPE
        DESCRIPTION.
    plane_vecs : TYPE
        DESCRIPTION.

    Returns
    -------
    point_proj : TYPE
        DESCRIPTION.

    '''
    # get plane normal
    normal = np.cross(plane_vecs[0,:],plane_vecs[1,:])
    
    # get point coordinates in the new basis, defined by the plane and its normal
    new_basis = np.concatenate((plane_vecs,normal[None,:]))
    point_new_coords = new_basis @ point
    
    norm_scale = point_new_coords[-1] # scale the normal by the point coordinate along that direction
    
    # get vector going from point to projection onto the plane
    # point_proj = point_new_coords - norm_scale*normal
    
    # point_proj = new_basis.T @ point_proj
    
    point_proj = point - norm_scale*normal
    
    return point_proj

def sortByPathLength(verts):
    if verts.shape[0]!=4:
        raise NotImplementedError()
        
    sorted_verts = np.zeros((verts.shape))
    n_verts = verts.shape[0]
    sorting_order = np.zeros((n_verts,),dtype=int)
    
    # get pairwise euclidean distances
    distances = euclidean_distances(verts)
    
    # find points with the largest distance between and set them to be opposite corners of a quadrilateral
    sorting_order[0] = np.where(distances==np.max(distances))[0][0]
    sorting_order[2] = np.where(distances==np.max(distances))[1][0]
                                    
    # sort the remaining indices by distances
    curr_point = sorting_order[0]
    possible_neighbours = np.setdiff1d(range(4),sorting_order[[0,2]])
    sorted_neighbours = np.argsort(distances[possible_neighbours,curr_point])
    sorting_order[1] = possible_neighbours[sorted_neighbours[0]]
    sorting_order[3] = possible_neighbours[sorted_neighbours[1]]
        
    for i in range(n_verts):
         sorted_verts[i,:] = verts[sorting_order[i],:]
    
    return sorted_verts, sorting_order


def sortByVecAngle(verts):
    if verts.shape[0]!=4:
        raise NotImplementedError()
    
    sorted_verts = np.zeros((verts.shape))
    n_verts = verts.shape[0]
    
    
    # calculate centre of mass
    com = np.mean(verts,axis=0)
    
    # express all verts as vectors going from com (normalise them too)
    # calculate the angles between vector corresponding to vertex 1 and all the others
    angles = []
    origin = np.zeros(verts[0].shape)
    v1 = np.stack((origin,verts[0,:]-com),axis=0)
    v1 /= np.linalg.norm(v1) # make it a unit vector
    
    for n in range(n_verts-1):
        v2 = np.stack((origin,verts[n+1,:]-com),axis=0)
        v2 /= np.linalg.norm(v2)
        
        cos_theta = np.dot(v1[1,:],v2[1,:])
        angles.append(cos_theta)
    
    angles = np.array(angles)
    
    sorting_order = np.concatenate((np.array([0]),np.argsort(angles)+1))
    sorted_verts = verts[sorting_order,:]
    
    return sorted_verts, sorting_order


def defPlaneShape(verts,plane, project= False):
    if verts.shape[0]!=4:
        raise NotImplementedError()
    
    # if project == True:
    #     proj = np.zeros(verts.shape)
    #     for i in range(verts.shape[0]):
    #         proj[i,:] = getProjection(verts[i,:],plane)
    # else:
    #     proj = verts
    hull = ConvexHull(verts) # fit a convex hull to the projected points
    #ignore the last coordinate - all points lie on the same plane and the function won't work if they're co-planar\
    
    sorting_order = hull.vertices
    convex_verts = verts[sorting_order,:]
    
    return convex_verts,sorting_order
    


# fig = plt.figure()
# ax = fig.add_subplot(231,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,1,2,3,0],0],proj[[0,1,2,3,0],1],proj[[0,1,2,3,0],2],'r-',alpha=0.2)

# # fig = plt.figure()
# ax = fig.add_subplot(232,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,1,3,2,0],0],proj[[0,1,3,2,0],1],proj[[0,1,3,2,0],2],'r-',alpha=0.2)

# # fig = plt.figure()
# ax = fig.add_subplot(233,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,2,1,3,0],0],proj[[0,2,1,3,0],1],proj[[0,2,1,3,0],2],'k-',alpha=0.2)

# # fig = plt.figure()
# ax = fig.add_subplot(234,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,2,3,1,0],0],proj[[0,2,3,1,0],1],proj[[0,2,3,1,0],2],'g-',alpha=0.2)
    
# # fig = plt.figure()
# ax = fig.add_subplot(235,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,3,1,2,0],0],proj[[0,3,1,2,0],1],proj[[0,3,1,2,0],2],'k-',alpha=0.2)
    
# # fig = plt.figure()
# ax = fig.add_subplot(236,projection='3d')
# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ko')
# ax.plot(proj[[0,3,2,1,0],0],proj[[0,3,2,1,0],1],proj[[0,3,2,1,0],2],'k-',alpha=0.2)
    
    
# #%%

# hull = ConvexHull(proj[:,:2])

# #%%
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot(verts[:,0],verts[:,1],verts[:,2],'ko')

# ax.plot(proj[:,0],proj[:,1],proj[:,2],'ro')


#     # plt.plot(proj[simplex,0],proj[simplex,1],proj[simplex,2],'r-')
# plt.plot(proj[hull.vertices,0],proj[hull.vertices,1],proj[hull.vertices,2],'k-')


    
# #%%

# plt.figure()
# square = np.array([[0,0,1,1],[0,1,1,0]])
# plt.plot(square,'o')
# square = square.T
# square = np.concatenate((square,np.zeros((square.shape[0],1))),1)
# # com = np.expand_dims(np.mean(square,1),1)
# # square = np.concatenate((square,com),1).T
# #%%

# import matplotlib.pyplot as plt

# hull = ConvexHull(square,'Qt')
# plt.plot(square[hull.vertices,0],square[hull.vertices,1],'r-')
# # plt.plot(square[hull.vertices[0],0],square[hull.vertices[0],1],'r-')




# scratch code - rotating vectors
# plt.figure()
# ax1 = plt.subplot(111, projection='3d')


# plane1_v1 = np.random.randn(3)
# plane1_v1 /= np.linalg.norm(plane1_v1)

# plane1_v2 = np.random.randn(3)
# plane1_v2 /= np.linalg.norm(plane1_v2)

# # orthogonalise the second vector
# if np.abs(np.dot(plane1_v1,plane1_v2))>0.00001:
#     plane1_v2 -= np.dot(plane1_v1,plane1_v2)*plane1_v1
#     plane1_v2 /= np.linalg.norm(plane1_v2)


# plane2_v1 = np.random.randn(3)
# plane2_v1 /= np.linalg.norm(plane2_v1)

# plane2_v2 = np.random.randn(3)
# plane2_v2 /= np.linalg.norm(plane2_v2)

# # orthogonalise the second vector
# if np.abs(np.dot(plane2_v1,plane2_v2))>0.00001:
#     plane2_v2 -= np.dot(plane2_v1,plane2_v2)*plane2_v1
#     plane2_v2 /= np.linalg.norm(plane2_v2)


# plane1_v1_v = vops.makeVec(plane1_v1)
# plane1_v2_v = vops.makeVec(plane1_v2)
# plane2_v1_v = vops.makeVec(plane2_v1)
# plane2_v2_v = vops.makeVec(plane2_v2)



# normal1 = np.cross(plane1_v1,plane1_v2)
# normal2 = np.cross(plane2_v1,plane2_v2)

# normal1_v = vops.makeVec(normal1)
# normal2_v = vops.makeVec(normal2)


# plt.figure()
# ax1 = plt.subplot(111, projection='3d')
# ax1.plot(plane1_v1_v[:,0],plane1_v1_v[:,1],plane1_v1_v[:,2],'r-')
# ax1.plot(plane1_v2_v[:,0],plane1_v2_v[:,1],plane1_v2_v[:,2],'r--',label='plane1')

# ax1.plot(plane2_v1_v[:,0],plane2_v1_v[:,1],plane2_v1_v[:,2],'k-')
# ax1.plot(plane2_v2_v[:,0],plane2_v2_v[:,1],plane2_v2_v[:,2],'k--',label='plane2')

# ax1.plot(normal1_v[:,0],normal1_v[:,1],normal1_v[:,2],'g-',label='normal1')
# ax1.plot(normal2_v[:,0],normal2_v[:,1],normal2_v[:,2],'g--',label='normal2')

# np.degrees(np.arccos(np.dot(normal1,normal2)))


# plt.figure()
# ax1 = plt.subplot(111, projection='3d')
# ax1.plot(normal1_v[:,0],normal1_v[:,1],normal1_v[:,2],'g-',label='normal1')
# ax1.plot(normal2_v[:,0],normal2_v[:,1],normal2_v[:,2],'g--',label='normal2')

# normal3 = np.cross(normal1,normal2)
# normal3_v = vops.makeVec(normal3)
# ax1.plot(normal3_v[:,0],normal3_v[:,1],normal3_v[:,2],'b-',label='normal3')


# np.degrees(np.arccos(np.dot(normal2,normal1)))

# triple_prod = np.dot(normal1,np.cross(normal2,normal3))
# triple_prod2 = np.dot(normal2,np.cross(normal1,normal3))



# v1 = np.array([1,0,0])
# v2 = np.array([.2,.9,0])
# v2 /= np.linalg.norm(v2)

# v1_v = vops.makeVec(v1)
# v2_v = vops.makeVec(v2)

# normal = np.cross(v1,v2)
# normal_v = vops.makeVec(normal)

# plt.figure()
# ax1 = plt.subplot(111, projection='3d')
# ax1.plot(v1_v[:,0],v1_v[:,1],v1_v[:,2],'g-',label='a')
# ax1.plot(v2_v[:,0],v2_v[:,1],v2_v[:,2],'b-',label='b')
# ax1.plot(normal_v[:,0],normal_v[:,1],normal_v[:,2],'k-',label='normal')

#         # if they're not already - rotate vector b 
#         # angle = np.arccos(np.dot(a,b))
        
#         # # a bit of a brute force approach here - try a positive angle first
#         # angle_diff = np.pi/2 - angle
#         # tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
        
#         # # check vectors orthogonal
#         # if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#         #     # if not - try rotating by the same angle but in the opposite direction
#         #     tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
        
#         # if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#         #     raise ValueError('New vectors still not orthogonal')
#         # else:
#         #     b_newBasis = tmp
        
#         # if they're not already - rotate vector b
#         # get directional angle between a and b
# triple_product = np.dot(v1,np.cross(v2,normal))

# if triple_product >= 0:
#     angle = np.arccos(np.dot(v1,v2))
# else:
#     angle = 2*np.pi - np.arccos(np.dot(a_newBasis,b_newBasis))

# if angle < np.pi/2:
#     angle_diff = np.pi/2 - angle
#     tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
# else:
#     angle_diff = angle - np.pi/2
#     tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)

# if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#     raise ValueError('New vectors still not orthogonal')
# else:
#     b_newBasis = tmp



