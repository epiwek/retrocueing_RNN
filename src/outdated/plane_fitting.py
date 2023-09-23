#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:07:16 2021
This file contains scripts used for testing the plane fitting methods.
@author: emilia
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from make_stimuli import make_stimuli
# from generate_data_gaussian import make_stimuli_gaussian
# from generate_data_1hot import make_stimuli_1hot

import vec_operations as vops
from rep_geom_analysis import get_best_fit_plane,rotate_plane_by_angle, \
align_plane_vecs,get_angle_between_planes_corrected, get_3D_coords, run_pca_pipeline
    
from subspace_alignment_index import get_simple_AI

from scipy.stats import mode, shapiro
#%%
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
        

def get_angle_between_planes_uncorrected(plane1_vecs, plane2_vecs):
    normal1 = get_normal(plane1_vecs)
    normal2 = get_normal(plane2_vecs)
    
    # since each normal has a length of 1, their dot product will be equal
    # to the cosine of the angle between them
    cos_theta = np.dot(normal1,normal2)
    theta = np.degrees(np.arccos(cos_theta))
    return theta


def get_normal(plane_vecs):
    normal = np.cross(plane_vecs[0,:],plane_vecs[1,:])
    return normal


def test_angle_correction(gt_angle,gt_phase,init_points=None):
    if np.all(init_points == None):
        # if initial plane shapes not suppllied, use squares as starting point
        plane_1_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
        plane_2_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T
    else:
        plane_1_datapoints = init_points[:4,:]
        plane_2_datapoints = init_points[4:,:]
    
    # rotate plane2 to get to gt_phase
    plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,gt_phase)
    
    # rotate plane2 to get to gt_angle
    plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints.T,
                                               gt_angle,axis='x').T
    
    points = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=0)
    
    plane1 = get_best_fit_plane(points[:4,:])
    plane2 = get_best_fit_plane(points[4:,:])
    
    print('Ground-truth angle : %.2f' %gt_angle)
    print('Ground-truth phase : %.2f' %gt_phase)
    
    angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
    print('Angle pre-correction: %.2f' %angle)
    
    angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
    print('Angle post-correction: %.2f' %angle_corrected)
    
    return angle_corrected
    

def test_phase_effect():
    '''
    Test if the gt phase has any effect on the estimation of the gt plane angle.
    Note plane angles are only defined on the [0,180] interval.

    Returns
    -------
    None.

    '''
    plt.figure()
    phases = np.arange(0,361,30)
    for gt_angle,c in zip(np.arange(0,181,30),sns.color_palette('husl',len(phases))):
        estimated_angles = [test_angle_correction(gt_angle,gt_phase) for gt_phase in phases]
        plt.plot(phases,estimated_angles,'o',color=c,label='gt angle = '+str(gt_angle))
    
    plt.xlabel('gt phase')
    plt.ylabel('estimated angle')
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()


def make_nonsq_geom(a1,b1,a2,b2):
    plane_1_datapoints = np.array([[a1,a1,-a1,-a1],[b1,-b1,-b1,b1],[0.5,0.5,0.5,0.5]])
    plane_2_datapoints = np.array([[a2,a2,-a2,-a2],[b2,-b2,-b2,b2],[-0.5,-0.5,-0.5,-0.5]])
    post = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=1).T
    
    return post
    

def test_nonsquare_effect():
    colours = ['r','y','g','b']

    a1,b2 = 1,1
    a2,b1 = 2,2
    init_points = make_nonsq_geom(a1,b1,a2,b2)
    angle_corrected = test_angle_correction(0,0,init_points)
    
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    plot_geometry(ax,init_points,colours)
    return angle_corrected
    

def make_nonrect_geom(n=4):
    '''
    Generates random non-rectangular geometries. No control over shape or 
    phase-alignment, but the planes are pallel.

    Parameters
    ----------
    n : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    None.

    '''
    plane_1_xys = gen_random_polygon(n)
    plane_2_xys = gen_random_polygon(n)
    
    plane1_zs = np.ones(n,)*.5
    plane2_zs = np.ones(n,)*-.5
    
    plane_1_datapoints = np.concatenate((plane_1_xys,plane1_zs[:,None]),axis=1)
    plane_2_datapoints = np.concatenate((plane_2_xys,plane2_zs[:,None]),axis=1)
    
    post = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=0)
    
    return post
    

def gen_random_polygon(n):
    '''
    Generates a random polygon in 2D with n vertices. X and Y coordinates 
    sampled from a uniform [-1,1] interval. Algorithm from:
        https://stackoverflow.com/questions/6758083/how-to-generate-a-random-convex-polygon
        Mangara's answer posted on Nov 17, 2017 at 20:22

    Parameters
    ----------
    n : number of vertices to generate
        DESCRIPTION.

    Returns
    -------
    vts : vertex coordinates
        DESCRIPTION.

    '''
    rng = np.random.default_rng()
    
    rnd_x = np.sort(2 * rng.random((n,)) -1)
    rnd_y = np.sort(2 * rng.random((n,)) -1)
    
    middle_ixs = np.arange(1,n-1)
    
    x_order = rng.permutation(middle_ixs)
    y_order = rng.permutation(middle_ixs)
    
    X1=np.concatenate((rnd_x[0,None],rnd_x[x_order[:len(middle_ixs)//2]],rnd_x[-1,None]))
    X2=np.concatenate((rnd_x[0,None],rnd_x[x_order[len(middle_ixs)//2:]],rnd_x[-1,None]))
    Y1=np.concatenate((rnd_y[0,None],rnd_y[y_order[:len(middle_ixs)//2]],rnd_y[-1,None]))
    Y2=np.concatenate((rnd_y[0,None],rnd_y[y_order[len(middle_ixs)//2:]],rnd_y[-1,None]))
    Xvec = np.concatenate((np.diff(X1),np.diff(np.flip(X2))))
    Yvec = np.concatenate((np.diff(Y1),np.diff(np.flip(Y2))))
                          
    yvec_ix = rng.permutation(len(Yvec))
    
    angles = [get_direction([i,j]) for i,j in zip(Xvec[yvec_ix],Yvec[yvec_ix])]
    
    angles_sort_ix = np.argsort(angles)
    
    vecs = np.stack((Xvec[yvec_ix][angles_sort_ix],
                           Yvec[yvec_ix][angles_sort_ix]),axis=1)
    
    vts = np.cumsum(vecs,0) + np.tile([rnd_x[0],rnd_y[0]],(n,1)) \
        + np.tile([rnd_x[-1],rnd_y[-1]],(n,1))
    
    return vts
    

def get_direction(vec):
    '''
    Get direction (angle) of a vector
    '''
    return np.degrees(np.arccos(np.dot(vec/np.linalg.norm(vec),[0,1])))


def test_shape_and_phase_effect(n_reps=100,gt_angle=0):
    '''
    Generate random quadrilateral geometries, not controlling for shape or 
    phase alignment, but holding the ground truth angle constant (
        default = 0 degrees).

    Parameters
    ----------
    n_reps : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    results = {}
    results['angles'] = np.empty((n_reps,))
    results['geoms'] = {}
    for i in range(n_reps):
        init_points = make_nonrect_geom(n=4)
        results['geoms'][str(i)]=init_points
        results['angles'][i]=test_angle_correction(gt_angle,0,init_points)
    
    plt.figure()
    plt.hist(results['angles'])
    plt.plot(np.ones(100,)*gt_angle,np.linspace(0,n_reps,100),'r-',label = 'gt angle')
    plt.legend()
    
    return results


def plot_pca_vecs(ax,plane1,plane2):
    '''
    Plot the plane-defining vectors obtained from pca.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plane1 : pca object for plane 1
        DESCRIPTION.
    plane2 : pca object for plane 2
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    all_vecs = np.concatenate((plane1.components_,plane2.components_),axis=0)
    #(x1,y1,x2,y2)
    pl = {'marker':['-','-','--','--'],
          'colour':['b','r','b','r'],
          'labels':['x1','y1','x2','y2']} # plot settings
    for i in range(len(all_vecs)):
        vec = vops.makeVec(all_vecs[i,:])
        ax.plot(vec[:,0],vec[:,1],vec[:,2],
                ls=pl['marker'][i],c=pl['colour'][i],label=pl['labels'][i])
    
    
    
    
    
# plt.figure()
# ax = plt.subplot(111,projection='3d')
# plot_geometry(ax,post,colours)
# plt.xlabel('x')

# plane1 = get_best_fit_plane(post[:4,:])
# plane2 = get_best_fit_plane(post[4:,:])

# angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
# print('Original/symmetric geometry')
# print('Angle pre-correction: %.2f' %angle)


from scipy.spatial import ConvexHull


def detect_concave_quadr(points_2D):
    if len(points) != 4:
        raise ValueError('4 vertices needed')
    hull = ConvexHull(points_2D)
    n_vertices = len(hull.vertices)
    is_concave = n_vertices < 4
    return is_concave
    
def detect_bowtie(points):

    n_points = len(points)
    # find plane vectors
    plane_vecs = get_best_fit_plane(points).components_
    # get projection onto a plane
    points_projected = \
        np.stack([getProjection(points[p,:],plane_vecs) for p in range(n_points)])
    #change_basis to that defined by plane vecs and normal
    points_newBasis = change_basis(points_projected)
    # get rid of the z-coord
    points_2D = points_newBasis[:,:2]
    
    # indices of the diagonal vertices
    diagonal_ixs = np.array([[0,2],[1,3]])
    # find the lines demarcated by the diagonals
    m1 = np.diff(points_2D[diagonal_ixs[0,:],1])/np.diff(points_2D[diagonal_ixs[0,:],0])
    m2 = np.diff(points_2D[diagonal_ixs[1,:],1])/np.diff(points_2D[diagonal_ixs[1,:],0])
    c1 = points_2D[diagonal_ixs[0,0],1] - m1*points_2D[diagonal_ixs[0,0],0] 
    c2 = points_2D[diagonal_ixs[1,0],1] - m2*points_2D[diagonal_ixs[1,0],0]
    
    # find the point where they cross
    x = (c2-c1)/(m1-m2)
    y = m1*x + c1
    
    # check if it lies between the ends of the two diagonals
    x_vs_d1 = np.logical_and(x >= points_2D[diagonal_ixs[0,:],0].min(),
                             x <= points_2D[diagonal_ixs[0,:],0].max())
    y_vs_d1 = np.logical_and(y >= points_2D[diagonal_ixs[0,:],1].min(),
                             y <= points_2D[diagonal_ixs[0,:],1].max())
    x_vs_d2 = np.logical_and(x >= points_2D[diagonal_ixs[1,:],0].min(),
                             x <= points_2D[diagonal_ixs[1,:],0].max())
    y_vs_d2 = np.logical_and(y >= points_2D[diagonal_ixs[1,:],1].min(),
                             y <= points_2D[diagonal_ixs[1,:],1].max())
    is_centre_within = np.all([x_vs_d1,x_vs_d2,y_vs_d1,y_vs_d2])
    
    # check if the datapoints can form a convex quadrilalteral
    is_concave = detect_concave_quadr(points_2D)
    
    is_bowtie = np.logical_and(not is_centre_within,is_concave)
    
    return is_bowtie
    

def test_ph_align(gt_angle,gt_phase,init_points,axis_align=0):
    if np.all(init_points == None):
        # if initial plane shapes not suppllied, use squares as starting point
        plane_1_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
        plane_2_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T
    else:
        plane_1_datapoints = init_points[:4,:]
        plane_2_datapoints = init_points[4:,:]
    
    # rotate plane2 to get to gt_phase
    plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,gt_phase)
    
    # rotate plane2 to get to gt_angle
    plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints.T,
                                               gt_angle,axis='x').T
    
    points = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=0)
    
    pa = phase_alignment_corr_ratio(points)
    r = pa['ratios'].mean()
    
    return r



def test_ph_align_phase_effect():
    '''
    Test how the correlation between colours changes with the gt phase, 
    holding the gt angle constant at 0.

    Returns
    -------
    None.

    '''
    plt.figure()
    phases = np.arange(0,361,5)
    r = [test_ph_align(0,gt_phase,None) for gt_phase in phases]
    plt.plot(phases,r,'k-o')
    
    plt.xlabel('gt phase')
    plt.ylabel('phase alignment')
    
    plt.tight_layout()

    return r


# def test_ph_align_axis_misaligned_phase_effect():
#     '''
#     Test how the correlation between colours changes with the gt phase, 
#     holding the gt angle constant at 0 for geometries that are not axis-aligned. 

#     Returns
#     -------
#     None.

#     '''
#     plt.figure()
#     phases = np.arange(0,361,5)
#     r = np.empty((len(phases)))
#     plane_1_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
#     for i,gt_phase in enumerate(phases):
#         plane_2_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T
#         plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,gt_phase).T
#         plane_1_datapoints = rotate_plane_by_angle(plane_1_datapoints,45,axis='x').T
#         plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,45,axis='x').T
#         points = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=0)
#         pa = phase_alignment_corr_ratio(points)
#         r[i] = pa['ratios'].mean()
#     plt.plot(phases,r,'k-o')
    
#     plt.xlabel('gt phase')
#     plt.ylabel('phase alignment')
    
    
#     plt.tight_layout()

#     return


def test_ph_align_angle_effect():
    '''
    Test how the correlation between colours changes with the gt angle, 
    holding the gt phase constant at 0.

    Returns
    -------
    None.

    '''
    plt.figure()
    angles = np.arange(0,181,5)
    r = [test_ph_align(gt_angle,0,None) for gt_angle in angles]
    plt.plot(angles,r,'k-o')
    
    plt.xlabel('gt angle')
    plt.ylabel('phase alignment')
    
    plt.tight_layout()

    return


# def test_phase_alignment_axis_misaligned_fullspace_r():
#     '''
#     Test how the correlation between colours changes with the gt phase.

#     Returns
#     -------
#     None.

#     '''
#     plt.figure()
#     phases = np.arange(0,361,5)
#     r = np.empty((len(phases)))
#     plane_1_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
    
#     for gt_angle,c in zip(np.arange(0,181,30),sns.color_palette('husl',len(phases))):
        
#         estimated_angles = [test_angle_correction(gt_angle,gt_phase) for gt_phase in phases]
#         plt.plot(phases,estimated_angles,'o',color=c,label='gt angle = '+str(gt_angle))
    
#     plt.xlabel('gt phase')
#     plt.ylabel('estimated angle')
    
#     plt.legend(bbox_to_anchor=(1,1))
#     plt.tight_layout()

    
#     for i,gt_phase in enumerate(phases):
#         plane_2_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T
#         plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,gt_phase).T
#         plane_1_datapoints = rotate_plane_by_angle(plane_1_datapoints,45,axis='x').T
#         plane_2_datapoints = rotate_plane_by_angle(plane_2_datapoints,45,axis='x').T
#         points = np.concatenate((plane_1_datapoints,plane_2_datapoints),axis=0)
#         pa = phase_alignment_corr(points)
#         r[i] = pa['r'].mean()
#     plt.plot(phases,r,'k-o')
    
#     plt.xlabel('gt phase')
#     plt.ylabel('phase alignment')
    
    
#     plt.tight_layout()

#     return

from scipy.stats import pearsonr, spearmanr
def phase_alignment_corr_ratio(data):
    
    n_colours = len(data)//2
    
    result = {'r':np.empty((n_colours,n_colours,2)),
              'p':np.empty((n_colours,n_colours,2)),
              'test':np.empty((n_colours,n_colours,2),dtype='str'),
              'ratios':np.empty((n_colours,2))}
    
    l1_centered = data[:n_colours,:] - data[:n_colours,:].mean(0)
    l2_centered = data[n_colours:,:] - data[n_colours:,:].mean(0)
    data_centered = np.concatenate((l1_centered,l2_centered),axis=0)
    print(data_centered.shape)
    
    for l in range(2):
        for n in range(n_colours):        
            c1_ix = n + l*n_colours
            c2_ix = np.arange((1-l)*n_colours,(2-l)*n_colours)
            
            for nn in range(n_colours):
                s1,p1 = shapiro(data_centered[c1_ix])
                s2,p2 = shapiro(data_centered[c2_ix[nn],:])
            
                if np.logical_and(p1>= .05, p2 >= .05):
                    result['test'][n,nn,l] = 'pearson'
                    result['r'][n,nn,l],result['p'][n,nn,l] = \
                        pearsonr(data_centered[c1_ix],data_centered[c2_ix[nn],:])
                else:
                    result['test'][n] = 'spearman'
                    result['r'][n,nn,l],result['p'][n,nn,l] = \
                        spearmanr(data_centered[c1_ix],data_centered[c2_ix[nn],:])  
            off_diag_ix = np.setdiff1d(np.arange(n_colours),n)         
            result['ratios'][n,l] = result['r'][n,n,l] / result['r'][n,off_diag_ix,l].mean()
    
    return result


def phase_alignment_corr(data):
    '''
    Calculate the correlation between population responses to the same colour 
    shown at different locations.

    Parameters
    ----------
    pca_data : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    
    n_colours = len(data)//2
    
    result = {'r':np.empty((n_colours,)),
              'p':np.empty((n_colours,)),
              'test':np.empty((n_colours,),dtype='str')}
    
    for n in range(n_colours):        
        l1_centered = data[n,:] - data[:n_colours,:].mean(0)
        l2_centered = data[n+n_colours,:] - data[n_colours:,:].mean(0)
        
        s1,p1 = shapiro(l1_centered)
        s2,p2 = shapiro(l2_centered)
        
        if np.logical_and(p1>= .05, p2 >= .05):
            result['test'][n] = 'pearson'
            result['r'][n],result['p'][n] = pearsonr(l1_centered,l2_centered)
        else:
            result['test'][n] = 'spearman'
            result['r'][n],result['p'][n] = spearmanr(l1_centered,l2_centered)   
    
    return result


def get_angle_test(points, plane1_vecs,plane2_vecs):
    normal1 = np.cross(plane1_vecs[0,:],plane1_vecs[1,:])
    normal2 = np.cross(plane2_vecs[0,:],plane2_vecs[1,:])
    
    normal3 = np.cross(normal1,normal2)
    normal3 /= np.linalg.norm(normal3)
    
    cos_theta = np.dot(normal1,normal2)
    sin_theta = np.linalg.det(np.stack((normal1,normal2,normal3)))
    
    # theta = np.degrees(np.arctan(sin_theta,cos_theta))
    
    return cos_theta,sin_theta


def plot_vec(ax,v,ls,c):
    v = vops.makeVec(v)
    ax.plot(v[:,0],v[:,1],v[:,2],ls=ls,c=c)



# def _rotmat(self, vector, points):
#     """
#     Rotates a 3xn array of 3D coordinates from the +z normal to an
#     arbitrary new normal vector.
#     """
    
#     vector /= np.linalg.norm(vector)
#     axis = np.cross([0,0,1], vector)
#     angle = np.degrees(np.arccos(np.dot(vg.angle(vg.basis.z, vector, units='rad')
    
#     a = np.hstack((axis, (angle,)))
#     R = matrix_from_axis_angle(a)
    
#     r = Rot.from_matrix(R)
#     rotmat = r.apply(points)
    
#     return rotmat


def construct_x_prod_matrix(u):
    u_x = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return u_x


def construct_rot_matrix(u,theta):
    u_x = construct_x_prod_matrix(u)
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*u_x + (1-np.cos(theta))*np.outer(u,u)
    return R


# def dir_angle_corr():
#     # the solution is the following:
#         # define a plane spanned by normal1 and normal2
#         # calculate the normal of that plane (normal3)
#         # determine the triple product : normal1 . (normal2 x normal3)
#         # if the above is positive - angle will be arccos(normal1 . normal2)
#         # otherwise, angle will be 360 - arccos(normal1 . normal2)
#     # normal3 = np.cross(normal1,normal2)
#     # triple_product = np.dot(normal1,np.cross(normal2,normal3))
    
#     # if triple_product >= 0:
#     #     angle_degrees = np.degrees(np.arccos(cos_theta))
#     # else:
#     #     angle_degrees = 360 - np.degrees(np.arccos(cos_theta))
#     angle_degrees = np.degrees(np.arccos(cos_theta))
#     return angle_degrees


def get_init_points():
    plane_1_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
    plane_2_datapoints = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T
    return plane_1_datapoints,plane_2_datapoints


def create_geom(init_points,gt_phase,gt_angle,gt_axis=0):
    if np.all(init_points == None):
        plane1_points, plane2_points = get_init_points()
        n_points = plane1_points.shape[0]
    else:
        n_points = init_points.shape[0]//2
        plane1_points, plane2_points = init_points[:n_points,:],init_points[n_points:,:]
        
    com1 = plane1_points.mean(0)
    com2 = plane2_points.mean(0)
    com = np.concatenate((np.tile(com1,(n_points,1)),
                    np.tile(com2,(n_points,1))),axis=0)
    plane1_points_centered = plane1_points - com1
    plane2_points_centered = plane2_points - com2
    
    # rotate plane2 to get to gt_phase
    plane2_points_centered = rotate_plane_by_angle(plane2_points_centered,gt_phase).T
    
    # rotate both planes by gt_axis_align 
    plane1_points_centered = rotate_plane_by_angle(plane1_points_centered,
                                                    gt_axis,axis='x').T
    plane2_points_centered = rotate_plane_by_angle(plane2_points_centered,
                                                    gt_axis,axis='x').T
    
    # rotate plane2 to get to gt_angle
    plane2_points_centered = rotate_plane_by_angle(plane2_points_centered,
                                               gt_angle,axis='x').T
    
    points_centered \
        = np.concatenate((plane1_points_centered,plane2_points_centered),axis=0)
    points = points_centered + com
    return points, points_centered


def make_coplanar(points_centered,normal1,normal2):
    # get the plane-defining vectors for both planes
    # this will be problematic if geometries distorted and mismatch in terms 
    # of the 'longest' side - need to port the x,y and normal directions from 
    # the angle correction function
    
    n_points = points_centered.shape[0]//2
    
    if np.logical_and(np.all(normal1==None),np.all(normal2==None)):
        plane1 = get_best_fit_plane(points_centered[:n_points,:])
        plane2 = get_best_fit_plane(points_centered[n_points:,:])
            
        #calculate the normals
        normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
        normal2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])
    
    # get the rotation axis and angle
    # if planes already co-planar, leave them as they are - need a workaround 
    # because cross-product is not defined then (i.e. it's 0)
    if np.abs(np.dot(normal1,normal2).round(6))==1:
        rot_angle=0
        rot_axis=[]
        R = np.eye(len(normal1))
    else:
        rot_angle = np.arccos(np.dot(normal1,normal2))
        # print(np.dot(normal1,normal2))
        if rot_angle > np.pi/2:
            rot_angle += np.pi
        
        # still to add here - sign of the angle
        rot_axis = np.cross(normal1,normal2)/np.linalg.norm(np.cross(normal1,normal2))
        
        # rot_vec = rot_angle * rot_axis
        # R = r.from_rotvec(rot_vec)
        # construct the rotation matrix
        R = construct_rot_matrix(rot_axis,rot_angle)
        
    # apply to plane2 points
    plane2_points = R.T @ points_centered[n_points:,:].T
    # plane2_points = R.apply(points_centered[n_points:,:],inverse=True)
    
    # get new coordinates
    points_new = np.concatenate((points_centered[:n_points,:],plane2_points.T),axis=0)
    # points_new = np.concatenate((points_centered[:n_points,:],plane2_points),axis=0)

    return points_new

    
def test_plane_rotations(init_points,gt_phase,gt_angle):
    # get plane1 and 2 initial datapoints (parallel planes)
    # if init_points == None:
    #     plane1_points, plane2_points = get_init_points()
    #     n_points = plane1_points.shape[0]
    # else:
    #     n_points = init_points.shape[0]//2
    #     plane1_points, plane2_points = init_points[:n_points,:],init_points[n_points:,:]
    
    # # # construct the desired geometry, characterised by a phse offset and between-plane angle
    # # points = create_geom(plane1_points,plane2_points,gt_angle,gt_phase)
    # # # get the centre of mass for each plane
    # # com = np.concatenate((np.tile(points[:n_points,:].mean(0),(n_points,1)),
    # #                 np.tile(points[n_points:,:].mean(0),(n_points,1))),axis=0)
    # # points_centered = points - com
    # com1 = plane1_points.mean(0)
    # com2 = plane2_points.mean(0)
    # com = np.concatenate((np.tile(com1,(n_points,1)),
    #                 np.tile(com2,(n_points,1))),axis=0)
    # plane1_points_centered = plane1_points - com1
    # plane2_points_centered = plane2_points - com2
    
    # points_centered = create_geom(plane1_points_centered,
    #                               plane2_points_centered,
    #                               gt_angle,gt_phase)
    
    
    points,points_centered = create_geom(init_points,gt_phase,gt_angle)
    com = points - points_centered
    n_points = len(points)//2
    points_centered = points_centered.round(3)
    # rotate one of the planes (2) to be co-planar with the other (1)
    points_new_centered = make_coplanar(points_centered,None,None)
    # fix numerical precision errors - for plotting
    
    points_new_centered = points_new_centered.round(3)
    points_new = points_new_centered + com
    
    #plot both
    plt.figure()
    colours = ['r','y','g','b']
    
    ax = plt.subplot(121,projection='3d')
    plot_geometry(ax,points,colours)
    
    plt.title('Gt phase %d, gt angle %.1f' %(gt_phase,gt_angle))
    # add the plane-defining vecs
    plane1 = get_best_fit_plane(points_centered[:n_points,:])
    plane2 = get_best_fit_plane(points_centered[n_points:,:])
    plot_pca_vecs(ax,plane1,plane2)
    
    ax2 = plt.subplot(122,projection='3d')
    plot_geometry(ax2,points_new_centered,colours)
    

    # d = get_pairwise_point_distance(points_new_centered) 
    # return d
    
# d = np.empty((91*2,))
# for ix,i in enumerate(np.arange(0,91,0.5)):
#     d[ix] = test_plane_rotations(None,0,i)
#     # plt.title('Gt angle = %d' %i)
# plt.figure()
# plt.plot(np.arange(0,91,0.5),d,'k-')

def get_pairwise_point_distance(points):
    n_points = len(points)//2
    d = [np.linalg.norm(points[i+n_points,:]-points[i,:]) for i in range(n_points)]
    return np.mean(d)


# def calc_phase_alignment(points):
#     pdb.set_trace()
#     n_points = len(points)//2
#     phases = np.empty((n_points,))
#     signs = np.empty((n_points,))
#     dots = np.empty((n_points,))
#     for n in range(n_points):
#         v1 = points[n,:] / np.linalg.norm(points[n,:])
#         v2 = points[n+n_points,:] / np.linalg.norm(points[n+n_points,:])
        
#         phases[n] = np.arccos(np.dot(v1,v2))
#         signs[n] = (1 if np.cross(v1,v2)[-1]>= 0 else -1)
#         dots[n] = np.dot(v1,v2)
#     is_mirror = check_if_mirror_imgs(signs,dots)
#     if is_mirror:
#         pa = np.nan 
#     else:
#         pa = (phases*signs).mean()
#     return np.degrees(pa)


def check_if_mirror_imgs(signs,dots):
    # characteristics of mirror image geometries
    # 1: normals between vectors of from the origin to the examined datapoints
    # point in opposite directions for hald of the examined datapoints
    if np.logical_and(signs.mean().round(6)==0,np.abs(signs).sum().round(6)!=0):
        is_mirror = True
    # 2: dot products point in opposite directions for half of the datapoints
    # if mean of dot products is ~ 0, it means they're pointing in the same
    # direction for two datapoints, and opposite for the other 2 - 
    # geometries are mirror images
    elif np.logical_and(dots.mean().round(6)==0,np.abs(dots).sum().round(6)!=0):
        is_mirror = True
    else:
        is_mirror = False
    return is_mirror


from vec_operations import getProjection



def test_phase_alignment(init_points,gt_phase,gt_angle):
    # get plane1 and 2 initial datapoints (parallel planes)
    # if init_points == None:
    #     plane1_points, plane2_points = get_init_points()
    #     n_points = plane1_points.shape[0]
    # else:
    #     n_points = init_points.shape[0]//2
    #     plane1_points, plane2_points = init_points[:n_points,:],init_points[n_points:,:]
    
    # # construct the desired geometry, characterised by a phse offset and between-plane angle
    # points = create_geom(plane1_points,plane2_points,gt_angle,gt_phase)
    # # get the centre of mass for each plane
    # com = np.concatenate((np.tile(points[:n_points,:].mean(0),(n_points,1)),
    #                 np.tile(points[n_points:,:].mean(0),(n_points,1))),axis=0)
    # points_centered = points - com
    # com1 = plane1_points.mean(0)
    # com2 = plane2_points.mean(0)
    # com = np.concatenate((np.tile(com1,(n_points,1)),
    #                 np.tile(com2,(n_points,1))),axis=0)
    # plane1_points_centered = plane1_points - com1
    # plane2_points_centered = plane2_points - com2
    
    # create the required geometry
    points,points_centered = create_geom(init_points,gt_phase,gt_angle)
    # plot
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    plot_geometry(ax,points,['r','y','g','b'])
    plt.title('Gt phase %d, gt angle %d' %(gt_phase,gt_angle))
    
    
    pa = get_phase_alignment(points)
    # plt.suptitle('Estimated phase %d' %pa.round())
    return pa


# from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.
    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.
    
    This is copied from the scipy.spatial package, the only change is that it 
    additionally returns the rotation matrix R
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
    R : array_like
        Rotation matrix that maps mtx1 to mtx2
    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.
    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets
    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.
    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.
    - The disparity scales as the number of points per input matrix.
    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
    Examples
    --------
    >>> from scipy.spatial import procrustes
    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:
    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0.0
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R,s


def change_basis(points):
    n_points = len(points)//2
    # find basis for plane 1
    plane1 = get_best_fit_plane(points[:n_points,:])
    normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
    
    new_basis = np.stack((plane1.components_[0,:],
                                plane1.components_[1,:],
                                normal1),axis=0).T
    # express datapoint coordinates in the new basis
    new_points = (new_basis.T @ points.T).T
    
    return new_points


def calc_phase_alignment(points_coplanar):
    n_points = len(points_coplanar)//2
    # change of basis to that defined by plane 1:
    # to zero-out the z-coordinate and reduce the problem to 2D
    points_new = change_basis(points_coplanar)
    # fix precision errors and throw out z-axis
    if points_new.round(6)[:,-1].sum() != 0:
        raise ValueError('Change of basis did not null the z-coordinates')
    points_new = points_new.round(6)[:,:2]
    # apply the procrustes analysis
    plane1_std, plane2_std, disparity,R,s = procrustes(points_new[:n_points,:], 
                                                    points_new[n_points:,:])
    
    reflection = np.linalg.det(R)<0
    bowtie1 = detect_bowtie(points_coplanar[:n_points,:])
    bowtie2 = detect_bowtie(points_coplanar[n_points:,:])
    if np.any(reflection,bowtie1,bowtie2):
        # estimating phase alignment doesn't make sense
        pa = np.nan
    else:
        pa = -np.degrees(np.arctan2(R[1,0],R[0,0]))
    
    return pa, reflection


def get_phase_alignment(points,normal1,normal2):
    n_points = len(points)//2
    
    # center datapoints
    plane1_points = points[:n_points,:] - points[:n_points,:].mean(0)
    plane2_points = points[n_points:,:] - points[n_points:,:].mean(0)
    
    # get plane_defining vecs from pca
    plane1_vecs = get_best_fit_plane(plane1_points).components_
    plane2_vecs = get_best_fit_plane(plane2_points).components_
    
    # project the datapoints onto their corresponding planes
    plane1_points_p = np.stack([getProjection(plane1_points[p,:],plane1_vecs) for p in range(n_points)])
    plane2_points_p = np.stack([getProjection(plane2_points[p,:],plane2_vecs) for p in range(n_points)])
    
    # center so that mean of each plane is at the origin
    plane1_points_c = plane1_points_p - plane1_points_p.mean(0)
    plane2_points_c = plane2_points_p - plane2_points_p.mean(0)
    
    all_points = np.concatenate((plane1_points_c,plane2_points_c),axis=0)
    # makes planes coplanar
    # by rotating plane 2 to be co-planar with plane 1
    points_coplanar = make_coplanar(all_points,normal1,normal2)
    
    # calculate the phase-alignment using procrustes analysis
    pa,reflection = calc_phase_alignment(points_coplanar)
    
    return pa


def get_phase_alignment_corrected(points):
    n_colours = len(points)//2
    points1 = points[:n_colours,:] # location 1 datapoints
    points2 = points[n_colours:,:] # location 2
    ixs_all = np.array([[0,-1,0,1],[1,0,1,2],[2,1,2,3],[3,2,3,0]]) 
    # indices of neighbouring sides of the parallelogram in the format 
    # [common vertex,x,common vertex,y]
    pa_all = np.empty((len(ixs_all),))
    
    # get plane_defining vecs from pca
    plane1_vecs = get_best_fit_plane(points1).components_
    plane2_vecs = get_best_fit_plane(points2).components_
    
    for combo in range(len(ixs_all)):
        plane_vecs_aligned1 = align_plane_vecs(points1,ixs_all[combo], plane1_vecs)
        plane_vecs_aligned2 = align_plane_vecs(points2,ixs_all[combo], plane2_vecs)

        # calculate normals for each plane
        normal1 = np.cross(plane_vecs_aligned1[0,:],plane_vecs_aligned1[1,:])
        normal2 = np.cross(plane_vecs_aligned2[0,:],plane_vecs_aligned2[1,:])
        
        pa_all[combo] = get_phase_alignment(points,normal1,normal2)
        
    pa = mode(pa_all)[0][0]   
    return pa


def get_plane_bases_corrected(points,plane1_vecs,plane2_vecs):
    # get the aligned plane vectors
    # indices of neighbouring sides of the data quadrilateral
    n_colours = len(points)//2
    dim = points.shape[1]
    points1 = points[:n_colours,:] # location 1 datapoints
    points2 = points[n_colours:,:] # location 2
    ixs_all = np.array([[0,-1,0,1],[1,0,1,2],[2,1,2,3],[3,2,3,0]]) 
    # indices of neighbouring sides of the parallelogram in the format 
    # [common vertex,x,common vertex,y]
    plane1_bases = np.empty((dim,dim,len(ixs_all)))
    plane2_bases = np.empty((dim,dim,len(ixs_all)))
    for combo in range(len(ixs_all)):
        plane_vecs_aligned1 = align_plane_vecs(points1,ixs_all[combo], plane1_vecs)
        plane_vecs_aligned2 = align_plane_vecs(points2,ixs_all[combo], plane2_vecs)

        # calculate normals for each plane
        normal1 = np.cross(plane_vecs_aligned1[0,:],plane_vecs_aligned1[1,:])
        normal2 = np.cross(plane_vecs_aligned2[0,:],plane_vecs_aligned2[1,:])
        
        plane1_bases[:,:,combo] = np.concatenate((plane_vecs_aligned1,normal1[None,:]),axis=0).T
        plane2_bases[:,:,combo] = np.concatenate((plane_vecs_aligned2,normal2[None,:]),axis=0).T
        
    return plane1_bases, plane2_bases



    
    
def get_angle_and_phase_between_planes_corrected(points,plane1_vecs,plane2_vecs):
    """
    Calculate the angle between two planes as each defined by two vectors.

    Parameters
    ----------
    plane1_vecs : numpy array
        plane1 vectors (from PCA, pre-aligment) as rows
        
    plane2_vecs : numpy array
        plane2 vectors (from PCA) as rows

    Returns
    -------
    angle_degrees: angle between planes in degrees
    pa_degrees: phase alignment between planes in degrees

    """
    # get the aligned plane vectors
    plane1_bases, plane2_bases = get_plane_bases_corrected(points,plane1_vecs,plane2_vecs)
    n_bases = plane1_bases.shape[-1]
    cos_theta_all = np.empty((n_bases,))
    sin_theta_all = np.empty((n_bases,))
    sign_theta_all = np.empty((n_bases,))
    pa_all = np.empty((n_bases,))
    
    n_points = len(points//2)
    for b in range(n_bases):
        
        # calculate the angle between planes
        normal1 = plane1_bases[:,-1,b]
        normal2 = plane2_bases[:,-1,b]
        
        # since each normal has a length of 1, their dot product will be equal
        # to the cosine of the angle between them
        cos_theta_all[b] = np.dot(normal1,normal2)
        # to get a *signed* angle, we also need the sin of the angle - which we 
        # can get from the normal of the two plane normals
        normal3 = np.cross(normal1,normal2)
        sin_theta_all[b] = np.linalg.det(np.stack((normal1,normal2,normal3)))
        
        # define the sign of the angle
        # this is an arbitrary distinction, but it is necessary to do circular 
        # statistics at the group level. Arccos will always map angles to the 
        # [0,180] range, whereas we want them to span the full circle. This 
        # rectification  will also mean that the mean angle across group will 
        # never be 0. Sign is determined based on z-coordinate of plane 1 normal
        # - if it's positive, so is the angle, otherwise - negative.
        
        # z1 = getProjection(points[0,:], plane1_bases[:,:2,b].T)[-1]
        # z2 = getProjection(points[n_points,:], plane2_bases[:,:2,b].T)[-1]
        
        # if z1-z2<=0:
        #     sign_theta_all[b] = 1
        # else:
        #     sign_theta_all[b] = -1
        
        if normal1[-1]>=0:
            sign_theta_all[b] = 1
        else:
            sign_theta_all[b] = -1
        
        
        # fix any numerical precision errors
        # cos_theta_all[b] = cos_theta_all[b].round(6)
        # sin_theta_all[b] = sin_theta_all[b].round(6)
        
        # calculate phase alignment
        # if angle between planes is within +-[90,180] range, it means that the 
        # planes are mirror images and calculating phase alignment does not 
        # make sense - set pa to nan
        if cos_theta_all[b] <= 0:
            pa_all[b] = np.nan
        else:
            pa_all[b] = get_phase_alignment(points,normal1,normal2)
            
    
    
    # get the plane angle - in degrees for convenience
    # angle_degrees_all = np.degrees(np.arccos(cos_theta_all))
    
    # angle_degrees_all = np.degrees(np.arctan2(sin_theta_all,cos_theta_all))
    
    # get the mode of all comparisons to estimate the final angle
    
    # #fix the numerical error if abs(np.dot) > 1
    # cos_theta_all[np.where(cos_theta_all > 1)] = 1
    # cos_theta_all[np.where(cos_theta_all < -1)] = -1
    
    # # get the plane angle - in degrees for convenience
    angle_degrees_all = np.degrees(np.arccos(cos_theta_all))
    angle_degrees_all *= sign_theta_all
    # get the mode of all comparisons to estimate the final angle and phase
    # this is necessary because some distortions of geometry, whereby the 
    # data quadrilateral is concave can give rise to an angle of 180 degrees
    # depending on the side of the quadrilateral that is used as the plane-defining
    # vector, even though the overall geometries are isometric w.r.t. to one 
    # another (and the ground truth). Taking the mode will remove this effect.
    angle_degrees = mode(angle_degrees_all)[0][0]    
    pa_degrees = mode(pa_all)[0][0] 
    return angle_degrees, pa_degrees



#%%


def test_angle_and_phase_alignment_steps(init_points,gt_phase,gt_angle,gt_axis,plot_data=False):
    # create the required geometry
    points,points_centered = create_geom(init_points,gt_phase,gt_angle,gt_axis)
    n_points = len(points)//2
    
    # make co-planar
    points_new = make_coplanar(points_centered,None,None)
    # fix precision errors
    points_new = points_new.round(6)
    
    # calculate the phase alignment from procrustes 
    plane1_std, plane2_std, disparity,R,s = procrustes(points_new[:n_points,:], 
                                                    points_new[n_points:,:])
    
    reflection = np.linalg.det(R)<0
    if reflection:
        # estimating phase alignment doesn't make sense
        pa = np.nan
    else:
        # r = Rot.from_matrix(R)
        # pa = -np.degrees(r.as_rotvec())[-1]
        pa = -np.degrees(np.arctan2(R[1,0],R[0,0]))
        
    
    
    if plot_data:
        # plot
        plt.figure()
        ax = plt.subplot(131,projection='3d')
        plot_geometry(ax,points_centered,['r','y','g','b'])
        plt.title('Gt phase %d, gt angle %d' %(gt_phase,gt_angle))
        
        ax2 = plt.subplot(132,projection='3d')
        plot_geometry(ax2,points_new,['r','y','g','b'])
        plt.title('Coplanar')
        
        ax3 = plt.subplot(133,projection='3d')
        plot_geometry(ax3,np.concatenate((plane1_std,plane2_std),axis=0),['r','y','g','b'])
        plt.title('Procrustes-aligned')
        
    return pa, reflection


def test_angle_and_phase_alignment_pipeline(init_points,gt_phase,gt_angle,gt_axis):
    # create the required geometry
    points,points_centered = create_geom(init_points,gt_phase,gt_angle,gt_axis)
    n_colours = len(points)//2
    
    plane1_vecs = get_best_fit_plane(points[:n_colours,:]).components_
    plane2_vecs = get_best_fit_plane(points[n_colours:,:]).components_
    angle, pa = \
        get_angle_and_phase_between_planes_corrected(points,plane1_vecs,plane2_vecs)
        
    return angle, pa

#%% test phase alignment with different gt phase and gt angle vals

plt.figure()
plt.ylim([-190,190])
angles = np.arange(1,181,10)
phases = np.arange(-180,181,30)
for gt_phase,c in zip(phases,sns.color_palette('husl',len(phases))):
    output = [test_angle_and_phase_alignment_pipeline(None,gt_phase,gt_angle,89) for gt_angle in angles]
    angles = np.array(output,dtype=object)[:,0]
    estimated_phases = np.array(output,dtype=object)[:,1]
    
    plt.plot(angles,estimated_phases,'o',color=c,label='gt phase = '+str(gt_phase),alpha=.3)

plt.xlabel('gt angle')
plt.ylabel('estimated phase')

plt.xlim((angles.min()-10,angles.max()+10))

plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
#%% ex1

colours = ['r','y','g','b']

post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T

post_down = rotate_plane_by_angle(post_down,45,axis='x').T
post = np.concatenate((post_up,post_down),axis=0)

plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

plane1 = get_best_fit_plane(post_up)
plane2 = get_best_fit_plane(post_down)

# get_angle_test(post, plane1.components_,plane2.components_)

# plot_pca_vecs(ax,plane1,plane2)


normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
normal2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])

normal3 = np.cross(normal1,normal2)/np.linalg.norm(np.cross(normal1,normal2))

plot_vec(ax,normal1,'-','g')
plot_vec(ax,normal2,'--','g')

plot_vec(ax,normal3,'-','m')


#%% rotate back

rot_angle = np.arccos(np.dot(normal1,normal2))
rot_axis = np.cross(normal1,normal2)/np.linalg.norm(np.cross(normal1,normal2))

R = construct_rot_matrix(rot_axis,rot_angle)

points_down = R @ post_down.T

points = np.concatenate((post_up,points_down.T),axis=0)
plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,points,colours)
#%% ex2


post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]]).T
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]]).T

post_down = rotate_plane_by_angle(post_down,90,axis='x').T
post = np.concatenate((post_up,post_down),axis=0)

plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

plane1 = get_best_fit_plane(post_down)
plane2 = get_best_fit_plane(post_up)

get_angle_test(post, plane1.components_,plane2.components_)

plot_pca_vecs(ax,plane1,plane2)


normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
normal2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])

# normal3 = np.cross(normal1,normal2)/np.linalg.norm(np.cross(normal1,normal2))

# plot_vec(ax,normal1,'-','g')
# plot_vec(ax,normal2,'--','g')

# plot_vec(ax,normal3,'-','m')


#%% original parallel plane geometry

colours = ['r','y','g','b']
n_colours=len(colours)


# pre_up = np.array([[1,1,-1,-1],[-1,1,1,-1],[0,0,0,0]])
# pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])

pre_up = np.array([[-1,-1,1,1],[1,-1,-1,1],[0,0,0,0]])
pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])

pre = np.concatenate((pre_up,pre_down),axis=1)

# post - both rotate

post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

post = np.concatenate((post_up,post_down),axis=1).T

#%%

# calculate the angle between
plane1 = get_best_fit_plane(post_up.T)
plane2 = get_best_fit_plane(post_down.T)

angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('Original/symmetric geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)

#%% change proportions to mismatch

post_up = np.array([[2,2,-2,-2],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[2,-2,-2,2],[-0.5,-0.5,-0.5,-0.5]])

post = np.concatenate((post_up,post_down),axis=1).T


plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

# calculate the best-fit planes
plane1 = get_best_fit_plane(post_up.T)
plane2 = get_best_fit_plane(post_down.T)

# add the plane vectors to the plot
plane1vec1 = vops.makeVec(plane1.components_[0,:])
plane1vec2 = vops.makeVec(plane1.components_[1,:])

plane2vec1 = vops.makeVec(plane2.components_[0,:])
plane2vec2 = vops.makeVec(plane2.components_[1,:])

normal1 = get_normal(plane1.components_)
normal2 = get_normal(plane2.components_)

normal1_vec = vops.makeVec(normal1)
normal2_vec = vops.makeVec(normal2)



ax.plot(plane1vec1[:,0],plane1vec1[:,1],plane1vec1[:,2],'r-',label='plane1vec1')
ax.plot(plane1vec2[:,0],plane1vec2[:,1],plane1vec2[:,2],'b-',label='plane1vec2')

ax.plot(plane2vec1[:,0],plane2vec1[:,1],plane2vec1[:,2],'r--',label='plane2vec1')
ax.plot(plane2vec2[:,0],plane2vec2[:,1],plane2vec2[:,2],'b--',label='plane2vec2')

ax.plot(normal1_vec[:,0],normal1_vec[:,1],normal1_vec[:,2],'k-',label='normal1')
ax.plot(normal2_vec[:,0],normal2_vec[:,1],normal2_vec[:,2],'k--',label='normal2')

plt.legend()

# calculate the angle between planes

angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('Mismatched proportions geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)


#%% align and plot plane vectors

# plane1vecs_aligned = align_plane_vecs(post_up.T, plane1)
# plane2vecs_aligned = align_plane_vecs(post_down.T, plane2)

# plane1vec1_aligned = vops.makeVec(plane1vecs_aligned[0,:])
# plane1vec2_aligned = vops.makeVec(plane1vecs_aligned[1,:])

# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])


# plt.figure()
# ax = plt.subplot(111,projection='3d')
# plot_geometry(ax,post.T,colours)


# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()

# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)


#%% test in-plane rotations 

# 90 degrees
post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

post_down = rotate_plane_by_angle(post_down.T,  90)

post = np.concatenate((post_up,post_down),axis=1).T


plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

# calculate planes
plane1 = get_best_fit_plane(post_up.T)
plane2 = get_best_fit_plane(post_down.T)


# plane1vecs_aligned = align_plane_vecs(post_up.T, plane1)
# plane2vecs_aligned = align_plane_vecs(post_down.T, plane2)


# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)


# calculate the angle between planes
angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('90 degree parallel rotation geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)


# plane1vec1_aligned = vops.makeVec(plane1vecs_aligned[0,:])
# plane1vec2_aligned = vops.makeVec(plane1vecs_aligned[1,:])

# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])
# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()


#%% 180 degrees

post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

post_down = rotate_plane_by_angle(post_down.T, 180)

post = np.concatenate((post_up,post_down),axis=1).T


plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

# calculate planes
plane1 = get_best_fit_plane(post_up.T)
plane2 = get_best_fit_plane(post_down.T)
# angle = get_angle_between_planes(plane1.components_, plane2.components_)
# print('Pre correction angle: %.2f' %angle)

# plane1vecs_aligned = align_plane_vecs(post_up.T, plane1)
# plane2vecs_aligned = align_plane_vecs(post_down.T, plane2)

# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)
# print('Post correction: %2f' %angle)


# plane1vec1_aligned = vops.makeVec(plane1vecs_aligned[0,:])
# plane1vec2_aligned = vops.makeVec(plane1vecs_aligned[1,:])

# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])
# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()


# n1vec = vops.makeVec(n1)
# n2vec = vops.makeVec(n2)
# ax.plot(n1vec[:,0],n1vec[:,1],n1vec[:,2],'k-',label='n1')
# ax.plot(n2vec[:,0],n2vec[:,1],n2vec[:,2],'k--',label='n2')

# calculate the angle between planes
angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('180 degree parallel rotation geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)

#%% mirror image
post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])

R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
post_flipped = np.concatenate((post_up,post_down),axis=1).T

post_flipped[n_colours:,:] = post_flipped[n_colours:,:]@R

plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post_flipped,colours)



# calculate planes
plane1 = get_best_fit_plane(post_flipped[:n_colours,:])
plane2 = get_best_fit_plane(post_flipped[n_colours:,:])
# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Pre correction angle: %.2f' %angle)


# plane1vecs_aligned = align_plane_vecs(post_flipped[:n_colours,:], plane1)
# plane2vecs_aligned = align_plane_vecs(post_flipped[n_colours:,:], plane2)

# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)
# print('Post correction: %2f' %angle)

# add the original plane vectors to the plot
# plane1vec1 = vops.makeVec(plane1.components_[0,:])
# plane1vec2 = vops.makeVec(plane1.components_[1,:])

# plane2vec1 = vops.makeVec(plane2.components_[0,:])
# plane2vec2 = vops.makeVec(plane2.components_[1,:])

# ax.plot(plane1vec1[:,0],plane1vec1[:,1],plane1vec1[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2[:,0],plane1vec2[:,1],plane1vec2[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1[:,0],plane2vec1[:,1],plane2vec1[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2[:,0],plane2vec2[:,1],plane2vec2[:,2],'b--',label='plane2vec2')

# plt.legend()


# plane1vec1_aligned = vops.makeVec(plane1vecs_aligned[0,:])
# plane1vec2_aligned = vops.makeVec(plane1vecs_aligned[1,:])

# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])
# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()


# calculate the angle between planes
angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('Mirror image geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post_flipped,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)
#%% try on non-rectangular planes

post_up = np.array([[1,1.5,-1,-1.5],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[1.5,-1,-1.5,1],[-0.5,-0.5,-0.5,-0.5]])

post = np.concatenate((post_up,post_down),axis=1).T


plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

# calculate planes
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])
# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Pre correction angle: %.2f' %angle)

# add the original plane vectors to the plot

# plane1vec1 = vops.makeVec(plane1.components_[0,:])
# plane1vec2 = vops.makeVec(plane1.components_[1,:])

# plane2vec1 = vops.makeVec(plane2.components_[0,:])
# plane2vec2 = vops.makeVec(plane2.components_[1,:])

# ax.plot(plane1vec1[:,0],plane1vec1[:,1],plane1vec1[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2[:,0],plane1vec2[:,1],plane1vec2[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1[:,0],plane2vec1[:,1],plane2vec1[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2[:,0],plane2vec2[:,1],plane2vec2[:,2],'b--',label='plane2vec2')

# plt.legend()


#% apply correction


# plane1vecs_aligned = align_plane_vecs(post[:n_colours,:], plane1)
# plane2vecs_aligned = align_plane_vecs(post[n_colours:,:], plane2)

# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)
# print('Post correction: %2f' %angle)

# plane1vec1_aligned = vops.makeVec(plane1vecs_aligned[0,:])
# plane1vec2_aligned = vops.makeVec(plane1vecs_aligned[1,:])

# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])
# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()

angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('Non-rectangular geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)

#%% test on non-rectangular and stretched out

post_up = np.array([[2,2.5,-2,-2.5],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
post_down = np.array([[1,1,-1,-1],[2.5,-2,-2.5,2],[-0.5,-0.5,-0.5,-0.5]])

post = np.concatenate((post_up,post_down),axis=1).T

plt.figure()
ax = plt.subplot(111,projection='3d')
plot_geometry(ax,post,colours)

# calculate planes
plane1 = get_best_fit_plane(post[:n_colours,:])
plane2 = get_best_fit_plane(post[n_colours:,:])
# angle = get_angle_between_planes(plane1.components_,plane2.components_)
# print('Pre correction angle: %.2f' %angle)

plane1vec1 = vops.makeVec(plane1.components_[0,:])
plane1vec2 = vops.makeVec(plane1.components_[1,:])

plane2vec1 = vops.makeVec(plane2.components_[0,:])
plane2vec2 = vops.makeVec(plane2.components_[1,:])

ax.plot(plane1vec1[:,0],plane1vec1[:,1],plane1vec1[:,2],'r-',label='plane1vec1')
ax.plot(plane1vec2[:,0],plane1vec2[:,1],plane1vec2[:,2],'b-',label='plane1vec2')

ax.plot(plane2vec1[:,0],plane2vec1[:,1],plane2vec1[:,2],'r--',label='plane2vec1')
ax.plot(plane2vec2[:,0],plane2vec2[:,1],plane2vec2[:,2],'b--',label='plane2vec2')

plt.legend()



#% apply correction

# plane1vecs_aligned = align_plane_vecs(post[:n_colours,:], plane1)
# plane2vecs_aligned = align_plane_vecs(post[n_colours:,:], plane2)

# angle = get_angle_between_planes(plane1vecs_aligned, plane2vecs_aligned)
# print('Post correction: %2f' %angle)

# plt.figure()
# ax = plt.subplot(111,projection='3d')
# plot_geometry(ax,post,colours)



# plane2vec1_aligned = vops.makeVec(plane2vecs_aligned[0,:])
# plane2vec2_aligned = vops.makeVec(plane2vecs_aligned[1,:])
# ax.plot(plane1vec1_aligned[:,0],plane1vec1_aligned[:,1],plane1vec1_aligned[:,2],'r-',label='plane1vec1')
# ax.plot(plane1vec2_aligned[:,0],plane1vec2_aligned[:,1],plane1vec2_aligned[:,2],'b-',label='plane1vec2')

# ax.plot(plane2vec1_aligned[:,0],plane2vec1_aligned[:,1],plane2vec1_aligned[:,2],'r--',label='plane2vec1')
# ax.plot(plane2vec2_aligned[:,0],plane2vec2_aligned[:,1],plane2vec2_aligned[:,2],'b--',label='plane2vec2')

# plt.legend()

angle = get_angle_between_planes_uncorrected(plane1.components_, plane2.components_)
print('Non-rectangular and mismatched proportions geometry')
print('Angle pre-correction: %.2f' %angle)

angle_corrected = get_angle_between_planes_corrected(post,plane1.components_,plane2.components_)
print('Angle post-correction: %.2f' %angle_corrected)

#%%

# N = len(pa_radians)

# radii, bin_edges = np.histogram(pa_radians[:,1],2)
# width = np.diff(bin_edges)[0]

# plt.figure()
# ax = plt.subplot(111, projection='polar')
# bars = ax.bar(bin_edges[:-1], radii, width=width, bottom=0.0)

# # Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor('b')
#     bar.set_alpha(0.5)

# plt.show()

plt.figure()
ax1 = plt.subplot(121, projection='polar')
radii = np.linspace(0,1,len(pa))
ax1.plot(pa_radians[:,0],radii,'o')

ax2 = plt.subplot(121, projection='polar')
ax2.plot(pa_radians[:,1],radii,'o')


#%% how sign of the angle is determined
import constants
angles = [45,135,-45,-135]

plt.figure()
for i,gt_angle in enumerate(angles):
    # create the geometry
    points, points_centered = create_geom(None,0,gt_angle)
    # # get the normals
    plane1 = get_best_fit_plane(points_centered[:4,:])
    plane2 = get_best_fit_plane(points_centered[4:,:])
    normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
    normal2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])

    
    # normal3 = np.cross(normal1,normal2)
    # normal3 /= np.linalg.norm(normal3)
    # plane1_basis = np.concatenate((plane1.components_,normal1[None,:]),axis=0).T
    # plane1_basis = plane1_basis[:,[1,0,-1]]
    
  
    
    # sign = get_plane_angle_sign(plane1_basis,normal2,np.dot(normal1,normal2))
    # print(sign)
    # plot
    ax = plt.subplot(2,2,i+1, projection='3d')
    plot_geometry(ax,points_centered,constants.PLOT_PARAMS['4_colours'])
    
    plot_vec(ax,normal1,'-','g')
    plot_vec(ax,normal2,'-','r')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Gt angle %d' %gt_angle)
    # new_normal2 = (plane1_basis.T @ normal2).T
    # print([new_normal2[-1]>=0,np.dot(normal1,normal2)<=0])
    # print(np.logical_xor(new_normal2[-1]>=0,np.dot(normal1,normal2)<=0))

    # def get_plane_angle_sign(plane1_basis,normal2,cos_theta):
    #     # get normal2 coordinates in the basis defined by plane 1 and its normal
    #     new_normal2 = (plane1_basis.T @ normal2).T
    #     sign = [1 if np.logical_xor(new_normal2[-1]>=0,np.dot(normal1,normal2)<=0) else -1]
    #     return sign
    
#%%
from numpy.linalg import norm
from numpy.random import randn

def gramSchmidt(n):
    '''
    Generate a random orthonormal matrix of dimensionality N.

    Parameters
    ----------
    n : int
        Dimensionality of the matrix.

    Returns
    -------
    Q : array (N,N)
        Orthonormal matrix of size (N,N). Columns equal to a set of orthogonal
        normalised unit vectors.

    '''
    
    
    # % function Q = gramSchmidt(N)
    # % Input: N - dimensionality of the basis
    # % Output: Q - NxN matrix with columns equal to a set of orthogonal
    # % normalised unit vectors
    
    Q = np.zeros((n,n))
    # generate the first basis vector
    v = randn(n)
    v = v/norm(v)
    Q[:,0] = v
    
    # generate the rest of basis vectors
    for i in range(1,n):
        v = randn(n)
        for ii in range(n-1):
            v -= np.dot(np.dot(v,Q[:,ii]),Q[:,ii])
        Q[:,i] = v/norm(v)
    
    # check that the matrix is orthonormal
    num_thr = 10e-10 # very small value to use as a numerical precision threshold
    if np.logical_or((Q@Q.T-np.eye(n)).sum() >= 10e-10, ((norm(Q,axis=1)-1).sum() >= num_thr)):
        raise ValueError('Matrix not orthonormal')
    return Q

# function Q = gramSchmidt(N)
#     % function Q = gramSchmidt(N)
#     % Input: N - dimensionality of the basis
#     % Output: Q - NxN matrix with columns equal to a set of orthogonal
#     % normalised unit vectors
    
#     % generate the first basis vector
#     Q = randn(N,1);
#     Q = Q./norm(Q);
    
#     % generate the rest of basis vectors
#     for n=2:N
#         v = randn(N,1);
#         for nn=1:n-1
#             v = v - v'*Q(:,nn)*Q(:,nn);
#         end
#         Q(:,n) = v./norm(v);
#     end
# end

#%%

# 1) generate some plane coordinates - should already have code for that
# 2) embed them in a higher-dim space
# 2.1 add noise for other dimensions (neurons) 
n_neurons = 200
n_noise_dims = n_neurons - 3

points, points_centered = create_geom(None,0,90,gt_axis=0)

# plot
plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,points_centered,constants.PLOT_PARAMS['4_colours'])

# plot_vec(ax,normal1,'-','g')
# plot_vec(ax,normal2,'-','r')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('Gt angle %d' %90)


noise_dims = np.random.randn(points.shape[0],n_noise_dims)/100
full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)

pca, coords_3D = get_3D_coords(full_dim_data)
plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,coords_3D,constants.PLOT_PARAMS['4_colours'])

# # 2.2 rotate everything in the full-D space (build some random rigid rotation matrix)
# R =  gramSchmidt(n_neurons)

# full_dim_data_rotated = full_dim_data @ R

subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
AI = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)



#%% test just AI vs different plane angles
angles = np.arange(0,181,5)
AI  = np.zeros((len(angles)))
 
for i,gt_angle in enumerate(angles):
    points, points_centered = create_geom(None,0,gt_angle,gt_axis=0)
    
    noise_dims = np.random.randn(points.shape[0],n_noise_dims)/100
    full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
    
    AI[i] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
plt.figure()
plt.plot(np.cos(np.radians(angles)),AI,'ko')
plt.ylim([0,1])
plt.xlabel('cos psi')
plt.ylabel('AI')
plt.title('Phase-aligned planes with known plane angle theta')
plt.tight_layout()



# test sensitivity to phase alignment - paralle planes
for i,gt_angle in enumerate(angles):
    points, points_centered = create_geom(None,gt_angle,0,gt_axis=0)
    
    noise_dims = np.random.randn(points.shape[0],n_noise_dims)/100
    full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
    
    AI[i] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
    
plt.figure()
plt.plot(np.cos(np.radians(angles)),AI,'ko')
plt.ylim([0,1])
plt.xlabel('cos psi')
plt.ylabel('AI')
plt.title('Paralell planes with known phase alignment angle psi')
plt.tight_layout()

# test sensitivity to phase alignment - orthogonal planes
for i,gt_angle in enumerate(angles):
    points, points_centered = create_geom(None,gt_angle,90,gt_axis=0)
    
    noise_dims = np.random.randn(points.shape[0],n_noise_dims)/100
    full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
    
    AI[i] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
    
plt.figure()
plt.plot(np.cos(np.radians(angles)),AI,'ko')
plt.ylim([0,1])
plt.xlabel('cos psi')
plt.ylabel('AI')
plt.title('90 degree planes with known phase alignment angle psi')
plt.tight_layout()

#%% test size of the planes
plane1points, plane2points =  get_init_points()
plane2points /= 4 # make plane 2 4x smaller than plane 1
 
points, points_centered = create_geom(np.concatenate((plane1points,plane2points),axis=0),
                                      gt_phase=0,gt_angle=0,gt_axis=0)

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,points_centered,constants.PLOT_PARAMS['4_colours'])


full_dim_data = np.concatenate((points_centered,np.zeros((points.shape[0],n_noise_dims))),axis=1)
full_dim_data = full_dim_data @ R
full_dim_data += np.random.randn(points.shape[0],n_neurons)/1000


# noise_dims = np.random.randn(points.shape[0],n_noise_dims)/1000
# full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)


pca, coords_3D = get_3D_coords(full_dim_data)

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,coords_3D,constants.PLOT_PARAMS['4_colours'])

subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
AI = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)

helpers.equal_axes(ax)

#%% run the above 100 times to see if parallel planes more disrupted by noise than orthogonal
n_neurons=200
n_draws = 100
AI, theta = np.zeros((n_draws,2)), np.zeros((n_draws,2))

for i in range(n_draws):
    R =  gramSchmidt(n_neurons)
    noise_dims = np.zeros((points.shape[0],n_noise_dims))
    for j,gt_angle in enumerate([0,90]):
        plane1points, plane2points =  get_init_points()
        plane2points /= 10 # make plane 2 4x smaller than plane 1
         
        points, points_centered = create_geom(np.concatenate((plane1points,plane2points),axis=0),
                                              gt_phase=0,gt_angle=gt_angle,gt_axis=0)
        
       
        full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
        full_dim_data = full_dim_data @ R
        full_dim_data += np.random.randn(points.shape[0],n_neurons)/1000
        
        subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
        AI[i,j] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
        theta[i,j] = subspace['theta']
    



AI_err_ortho = (0.5-AI[:,1])
AI_err_parallel = (1-AI[:,0])

run_contrast_single_sample(AI_err_ortho-AI_err_parallel,[0],alt='two-sided')

theta_err_ortho = (90-np.abs(theta[:,1]))
theta_err_parallel = (0-np.abs(theta[:,0]))
angle_diff_radians = np.radians(theta_err_ortho - theta_err_parallel)
p_post, v_post = pycircstat.tests.vtest(angle_diff_radians, np.radians(0))
    
#%% test AI on non-noisy orthogonal planes
plane1points, plane2points =  get_init_points()
# plane2points /= 4 # make plane 2 4x smaller than plane 1
 
points, points_centered = create_geom(np.concatenate((plane1points,plane2points),axis=0),
                                      gt_phase=0,gt_angle=90,gt_axis=0)

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,points_centered,constants.PLOT_PARAMS['4_colours'])


full_dim_data = np.concatenate((points_centered,np.zeros((points.shape[0],n_noise_dims))),axis=1)
full_dim_data = full_dim_data @ R
# full_dim_data += np.random.randn(points.shape[0],n_neurons)/100


# noise_dims = np.random.randn(points.shape[0],n_noise_dims)/1000
# full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)


pca, coords_3D = get_3D_coords(full_dim_data)

plt.figure()
ax = plt.subplot(111, projection='3d')
plot_geometry(ax,coords_3D,constants.PLOT_PARAMS['4_colours'])
helpers.equal_axes(ax)

subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
AI = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)



#%% get the gt AI values for known plane angles

def get_AI_gt(n_neurons,gt_angles,n_datapoints,n_noise_dims):
    
    
    AI_gt = np.zeros((len(gt_angles)))
    # R =  gramSchmidt(n_neurons)
    noise_dims = np.zeros((n_datapoints,n_noise_dims))
    
    for j,gt_angle in enumerate(gt_angles):
        points, points_centered = create_geom(None,gt_phase=0,gt_angle=gt_angle,gt_axis=0)
        full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
        # full_dim_data = full_dim_data @ R - not necessary
        AI_gt[j] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
    return AI_gt
        

#%% test how plane fitting angle estimation is affected by noise when planes
# have highly unequal areas

n_neurons=200
n_noise_dims = n_neurons - 3

n_draws = 100
plane_2_scaling = 1/np.array([1.1,2,4,6,8,10,12,14,16,18])

n_conditions = len(plane_2_scaling)

n_datapoints = 8

gt_angles = np.arange(0,91,10)
gt_AIs = get_AI_gt(n_neurons,gt_angles,n_datapoints,n_noise_dims)

n_gt_angles = len(gt_angles)

AI, theta = np.zeros((n_conditions,n_draws,n_gt_angles)), np.zeros((n_conditions,n_draws,n_gt_angles))

for s,scale in enumerate(plane_2_scaling):
    for i in range(n_draws):
        R =  gramSchmidt(n_neurons)
        noise_dims = np.zeros((n_datapoints,n_noise_dims))
        
        for j,gt_angle in enumerate(gt_angles):
            plane1points, plane2points =  get_init_points()
            plane1points *= scale # make plane 2 smaller than plane 1 by scale factor
             
            points, points_centered = create_geom(np.concatenate((plane1points,plane2points),axis=0),
                                                  gt_phase=0,gt_angle=gt_angle,gt_axis=0)
            
           
            full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
            full_dim_data = full_dim_data @ R
            full_dim_data += np.random.randn(points.shape[0],n_neurons)/1000
            
            subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
            AI[s,i,j] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
            theta[s,i,j] = subspace['theta']
        
    
#%% plot

theta_err_st = np.zeros((n_gt_angles,len(plane_2_scaling)))
AI_err_st = np.zeros((n_gt_angles,len(plane_2_scaling)))

for ix in range(n_gt_angles):
    gt_angle = gt_angles[ix]
    gt_AI = gt_AIs[ix]
    
    
    theta_err = np.abs(gt_angle - np.abs(theta[:,:,ix]))
    AI_err = np.abs(gt_AI - AI[:,:,ix])




    # plt.figure()
    # plt.plot(plane_2_scaling,AI_err.mean(-1),'b-o')
    
    
    # plt.figure()
    # plt.plot(plane_2_scaling,np.abs(theta_err).mean(-1),'r-o')
    
    
    theta_err_st[ix,:] = (theta_err.mean(-1) - theta_err.mean(-1)[0]) / theta_err.mean(-1)[-1]
    AI_err_st[ix,:] = (AI_err.mean(-1) - AI_err.mean(-1)[0]) / AI_err.mean(-1)[-1]
    
    plt.figure()
    plt.plot(plane_2_scaling,theta_err_st[ix,:],'r--o',label='theta')
    plt.plot(plane_2_scaling,AI_err_st[ix,:],'b-o',label='AI')
    
    plt.xlabel('Plane 2 scaling factor')
    plt.ylabel('Standardised estimation error')
    
    plt.title('Gt angle = %d' %gt_angle)
    
    plt.tight_layout()




# plot average across all gt angles
plt.figure()
# plt.plot(plane_2_scaling,theta_err_st.mean(0),'r--o',label='theta')
# plt.plot(plane_2_scaling,AI_err_st.mean(0),'b-o',label='AI')


plt.errorbar(plane_2_scaling,theta_err_st.mean(0),c='r',label='theta',yerr=theta_err_st.std(0))
plt.errorbar(plane_2_scaling,AI_err_st.mean(0),c='b',label='AI',yerr=AI_err_st.std(0))



plt.xlabel('Plane 2 scaling factor')
plt.ylabel('Standardised estimation error')




#%% show how good plane angle estimation is when planes have equal sizes


n_neurons=200
n_noise_dims = n_neurons - 3

n_draws = 100


n_datapoints = 8

gt_angles = np.arange(0,91,10)
gt_AIs = get_AI_gt(n_neurons,gt_angles,n_datapoints,n_noise_dims)

n_gt_angles = len(gt_angles)

AI_baseline, theta_baseline = np.zeros((n_draws,n_gt_angles)), np.zeros((n_draws,n_gt_angles))
AI_err_baseline, theta_err_baseline = np.zeros((n_draws,n_gt_angles)), np.zeros((n_draws,n_gt_angles))

for i in range(n_draws):
    R =  gramSchmidt(n_neurons)
    noise_dims = np.zeros((n_datapoints,n_noise_dims))
    
    for j,gt_angle in enumerate(gt_angles):
        plane1points, plane2points =  get_init_points()
         
        points, points_centered = create_geom(None,gt_phase=0,gt_angle=gt_angle,gt_axis=0)
        
       
        full_dim_data = np.concatenate((points_centered,noise_dims),axis=1)
        full_dim_data = full_dim_data @ R
        full_dim_data += np.random.randn(points.shape[0],n_neurons)/1000
        
        subspace = run_pca_pipeline(constants,full_dim_data,['subspace1',['subspace2']])
        AI_baseline[i,j] = get_simple_AI(full_dim_data[:4,:],full_dim_data[4:,:],2)
        theta_baseline[i,j] = subspace['theta']
    
        theta_err_baseline[i,j] = gt_angle - np.abs(theta_baseline[i,j])
        AI_err_baseline[i,j] = gt_AIs[j] - AI_baseline[i,j]

# plt.plot(plane_2_scaling,theta_err_st.mean(0),'r--o',label='theta')
# plt.plot(plane_2_scaling,AI_err_st.mean(0),'b-o',label='AI')


plt.figure()
plt.errorbar(gt_angles,theta_err_baseline.mean(0),c='r',label='theta',yerr=theta_err_baseline.std(0))
plt.xlabel('Gt angle')
plt.ylabel('Estimation error (degrees)')
plt.tight_layout()

plt.figure()
plt.errorbar(gt_angles,AI_err_baseline.mean(0),c='b',label='AI',yerr=AI_err_baseline.std(0))
plt.xlabel('Gt angle')
plt.ylabel('Estimation error (AI)')
plt.tight_layout()






