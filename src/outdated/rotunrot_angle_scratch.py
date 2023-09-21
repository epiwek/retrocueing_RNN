#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:16:18 2023

@author: emilia
"""
import helpers
from rep_geom_analysis import run_pca_pipeline





#%% same but project everything into common 3D space

def get_plane_angle_cued_across_delays_common_space(params,cv=2):
    '''
    For each location, get the AI between its subspaces from the 
    pre- and post-cue delay intervals. Do it in 2-fold cross-validation to 
    determine if both or only one of the pre-cue subspaces are are rotated post-cue
    to form a parallel plane geometry.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    AI_tbl : array
        AI values for the 'unrotated' and 'rotated' planes, averaged across cv 
        folds. Format: (n_dims,(unrotated,rotated),model)
    same_ixs : array
        Indexes of the unrotated plane for each model.
    trial_ixs : dict
        Train and test trial indexes for the cross-validation folds.

    '''
    load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    max_dim = 3
    n_dims = max_dim-1
    
    AI_table_ad_train = np.zeros((n_dims,2,params['n_models'],cv))
    AI_table_ad_test = np.zeros((n_dims,2,params['n_models'],cv))
    
    theta_ad_train = np.zeros((2,params['n_models'],cv))
    theta_ad_test = np.zeros((2,params['n_models'],cv))
    
    psi_ad_train = np.zeros((2,params['n_models'],cv))
    psi_ad_test = np.zeros((2,params['n_models'],cv))
    
    same_ixs_AI = np.zeros((2,params['n_models'],2))
    same_ixs_theta = np.zeros((params['n_models'],2))
    
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1
    
    trial_ixs = {'train':{},'test':{}}
    
    PVE_3D = np.zeros((2,2,params['n_models'])) # cv folds, unrotated/rotated, model
    
    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        train,test = get_trial_ixs(params,cv=2)
        trial_ixs['train'][model_number],trial_ixs['test'][model_number] = \
            train, test
        
        
        loc_split = params['B']
        for i in range(2):
            # bin the train and test datasets into colour bins
            data_train = helpers.bin_data(params, eval_data['data'][train[i],:,:])
            data_test = helpers.bin_data(params, eval_data['data'][test[i],:,:])
            
            delay1_train = data_train[:,d1_ix,:] 
            delay1_test = data_test[:,d1_ix,:] - data_test[:,d1_ix,:].mean()
            
            delay2_train = data_train[:,d2_ix,:] - data_train[:,d2_ix,:].mean()
            delay2_test = data_test[:,d2_ix,:] - data_test[:,d2_ix,:].mean()
            
            # calculate the AI on train data 
            # for dim in range(2,max_dim+1):
            #     AI_table_ad_train[dim-2,0,model,i]=get_simple_AI(delay1_train[:loc_split,:],
            #                                                   delay2_train[:loc_split,:],max_dim) #loc1
            #     AI_table_ad_train[dim-2,1,model,i]=get_simple_AI(delay1_train[loc_split:,:],
            #                                                   delay2_train[loc_split:,:],max_dim) #loc2
            
            # # find the plane that stays the same
            # same_ixs_AI[:,model,i] = np.argmax(AI_table_ad_train[:,:,model,i],1)
            
            
            
            # # calculate the plane angles theta on train data
            # loc1 cued
            # loc1_subspace = run_pca_pipeline(constants,
            #                   torch.cat((delay1_train[:loc_split,:],
            #                                                   delay2_train[:loc_split,:])),
            #                   ['pre_delay','post_delay']) #loc1
            # # loc 2 cued
            # loc2_subspace = run_pca_pipeline(constants,
            #                   torch.cat((delay1_train[loc_split:,:],
            #                                                   delay2_train[loc_split:,:])),
            #                   ['pre_delay','post_delay']) #loc2
            
            # same_ixs_theta[model,i] = np.argmax([np.abs(np.cos(np.radians(loc1_subspace['theta']))),
            #                                   np.abs(np.cos(np.radians(loc2_subspace['theta'])))])
            
            # pre-cue loc1 cued, post-cue loc1 cued, pre-ue loc2 cued, post-cue loc2 cued
            data1 = torch.cat((delay1_test[:loc_split,:],
                                            delay1_test[loc_split:,:]))
            data1 -= data1.mean()                                
            
            data2 = torch.cat(( delay2_test[:loc_split,:],
                              delay2_test[loc_split:,:]))
            
            data2 -= data2.mean()  
            
            pca, coords_3D = get_3D_coords(torch.cat((data1[:loc_split,:],
                                            data2[:loc_split,:],
                                            data1[loc_split:,:],
                                            data2[loc_split:,:])))
            
            # get loc1 angle
            # get loc2 angle
            
            plane1_1 = get_best_fit_plane(coords_3D[:n_colours,:])
            plane1_2 = get_best_fit_plane(coords_3D[n_colours:n_colours*2,:])
            plane2_1 = get_best_fit_plane(coords_3D[n_colours*2:n_colours*3,:])
            plane2_2 = get_best_fit_plane(coords_3D[n_colours*3:n_colours*4,:])
            
            # get angle between planes and phase alignment
            # theta = get_angle_between_planes_corrected(coords_3D,
            #                                            plane1.components_,
            #                                            plane2.components_)
            
            theta_loc1, pa_loc1 = get_angle_and_phase_between_planes_corrected(coords_3D[:8,:],
                                                        plane1_1.components_,
                                                        plane1_2.components_)
            
            theta_loc2, pa_loc2 = get_angle_and_phase_between_planes_corrected(coords_3D[8:,:],
                                                        plane2_1.components_,
                                                        plane2_2.components_)
            
            # find the plane that stays the 'same' - abs cos of theta will be larger
            same_ixs_theta[model,i] = np.argmax([np.abs(np.cos(np.radians(theta_loc1))),
                                              np.abs(np.cos(np.radians(theta_loc2)))])
            
            
            if same_ixs_theta[model,i] == 0:
                stay_plane_ix = np.arange(loc_split)
            else:
                stay_plane_ix = np.arange(loc_split,loc_split*2)
            switch_plane_ix = np.setdiff1d(np.arange(loc_split*2),stay_plane_ix)
               
            
            # first, check the angles on AI-defined 'rotated' and 'unrotated' planes
            unrotated = torch.cat((delay1_test[stay_plane_ix,:],
                                                          delay2_test[stay_plane_ix,:]),dim=0)
            rotated = torch.cat((delay1_test[switch_plane_ix,:],
                                                          delay2_test[switch_plane_ix,:]),dim=0)


            # unrotated_subspace = run_pca_pipeline(constants,
            #                   unrotated,
            #                   ['pre_delay','post_delay'])
            # rotated_subspace = run_pca_pipeline(constants,
            #                   rotated,
            #                   ['pre_delay','post_delay'])
            
            # theta_ad_test[0,model,i] =  unrotated_subspace['theta']
            # theta_ad_test[1,model,i] =  rotated_subspace['theta']
            
            
            
            # psi_ad_test[0,model,i] =  unrotated_subspace['psi']
            # psi_ad_test[1,model,i] =  rotated_subspace['psi']
            
            data1 = torch.cat((delay1_test[stay_plane_ix,:],
                                            delay1_test[switch_plane_ix,:]))
            data1 -= data1.mean()                                
            
            data2 = torch.cat(( delay2_test[stay_plane_ix,:],
                              delay2_test[switch_plane_ix,:]))
            
            data2 -= data2.mean()  
            
            pca_test, coords_3D_test = get_3D_coords(torch.cat((data1[:4,:],
                                            data2[:4,:],
                                            data1[4:,:],
                                            data2[4:,:])))
            
            
            
            # pca_test, coords_3D_test = get_3D_coords(torch.cat((delay1_test[stay_plane_ix,:],
            #                                 delay2_test[stay_plane_ix,:],
            #                                 delay1_test[switch_plane_ix,:],
            #                                 delay2_test[switch_plane_ix,:])))
            
            unrot_plane1 = get_best_fit_plane(coords_3D_test[:n_colours,:])
            unrot_plane2 = get_best_fit_plane(coords_3D_test[n_colours:n_colours*2,:])
            rot_plane1 = get_best_fit_plane(coords_3D_test[n_colours*2:n_colours*3,:])
            rot_plane2 = get_best_fit_plane(coords_3D_test[n_colours*3:n_colours*4,:])
            
            theta_ad_test[0,model,i], psi_ad_test[0,model,i] = get_angle_and_phase_between_planes_corrected(coords_3D[:8,:],
                                                        unrot_plane1.components_,
                                                        unrot_plane2.components_)
            
            theta_ad_test[1,model,i], psi_ad_test[1,model,i] = get_angle_and_phase_between_planes_corrected(coords_3D[8:,:],
                                                        rot_plane1.components_,
                                                        rot_plane2.components_)
            
            # PVE_3D[i,0,model] = unrotated_subspace['pca'].explained_variance_ratio_.sum()
            # PVE_3D[i,1,model] = rotated_subspace['pca'].explained_variance_ratio_.sum()
            
            # calculate precue subspace
            # precue_subspace = run_pca_pipeline(constants,
            #                   torch.cat((delay1_test[stay_plane_ix,:],
            #                                                   delay1_test[switch_plane_ix,:]),dim=0),
            #                   ['unrotated','rotated'])
            
            # quick_plotter(model,unrotated_subspace,rotated_subspace)
            
            # calculate the AI on test data

            
            # for dim in range(2,max_dim+1):
            #     if same_ixs_AI[dim-2,model,i] == 0:
            #         stay_plane_ix = np.arange(loc_split)
            #     else:
            #         stay_plane_ix = np.arange(loc_split,loc_split*2)
            #     switch_plane_ix = np.setdiff1d(np.arange(loc_split*2),stay_plane_ix)
                
            #     AI_table_ad_test[dim-2,0,model,i]=get_simple_AI(delay1_test[stay_plane_ix,:],
            #                                                   delay2_test[stay_plane_ix,:],dim) #stay plane
            #     AI_table_ad_test[dim-2,1,model,i]=get_simple_AI(delay1_test[switch_plane_ix,:],
            #                                                   delay2_test[switch_plane_ix,:],dim) #switch plane
            
            
    # get averages across cv splits
    theta_unrotated_mean = np.array([theta_ad_test[0,m,:].mean(-1) if np.sign(theta_ad_test[0,m,:]).sum()!=0 else np.abs(theta_ad_test[0,m,:]).mean(-1) for m in range(params['n_models'])])
    theta_rotated_mean = np.array([theta_ad_test[1,m,:].mean(-1) if np.sign(theta_ad_test[1,m,:]).sum()!=0 else np.abs(theta_ad_test[1,m,:]).mean(-1) for m in range(params['n_models'])])
    psi_unrotated_mean = np.array([psi_ad_test[0,m,:].mean(-1) if np.sign(psi_ad_test[0,m,:]).sum()!=0 else np.abs(psi_ad_test[0,m,:]).mean(-1) for m in range(params['n_models'])])
    psi_rotated_mean = np.array([psi_ad_test[1,m,:].mean(-1) if np.sign(psi_ad_test[1,m,:]).sum()!=0 else np.abs(psi_ad_test[1,m,:]).mean(-1) for m in range(params['n_models'])])
            
    plot_AI(params,AI_table_ad_test.mean(-1),'unrotrot','cued')

            
    return AI_table_ad_test.mean(-1), same_ixs, trial_ixs


#%% plot

# plot theta unroated and rotated
angles_radians = np.radians(np.stack((theta_unrotated_mean,theta_rotated_mean),1))
plot_plane_angles_multiple(constants,angles_radians,r=None,paired=False,cu=True,expt3=False,custom_labels=['unrotated','rotated'])


# plot_plane_angles_single(constants,np.radians(theta_unrotated_mean),'cu',r=None)
# plot_plane_angles_single(constants,np.radians(theta_rotated_mean),'cu',r=None)

# plot the difference rotated-unrotated
rot_unrot_dist = np.abs(theta_rotated_mean)-np.abs(theta_unrotated_mean)
plot_plane_angles_single(constants,np.radians(rot_unrot_dist),'cu',r=None)
# plot_plane_angles_multiple(constants,angles_radians,r=None,paired=True,cu=False,expt3=False)


# angles_radians = np.radians(np.stack((theta_unrotated_mean,rot_unrot_dist),1))

# plot pa
angles_radians = np.radians(np.stack((psi_unrotated_mean,psi_rotated_mean),1))
plot_plane_angles_multiple(constants,angles_radians,r=None,paired=False,cu=True,expt3=False,custom_labels=['unrotated','rotated'])

# angles_radians = np.radians(np.stack((theta_unrotated_mean,theta_rotated_mean),1))
# plot_plane_angles_multiple(constants,angles_radians,r=None,paired=False,cu=False,expt3=False)

def plot_rot_unrot_geom(precue_subspace,unrotated_subspace,rotated_subspace):
    plt.figure()
    # plot both pre-cue
    
    ax = plt.subplot(1,3,1, projection='3d')
    plot_geometry(ax,precue_subspace['3Dcoords'],precue_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels=precue_subspace['binned_data_subspace_order'])
    ax.set_title('Pre-cue, Theta = %.2f, PVE = %.2f' %(precue_subspace['theta'],precue_subspace['pca'].explained_variance_ratio_.sum()))
    
    # plot unrotated
    ax2 = plt.subplot(1,3,2, projection='3d')
    plot_geometry(ax2,unrotated_subspace['3Dcoords'],unrotated_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels=unrotated_subspace['binned_data_subspace_order'])
    ax2.set_title('Unrotated, Theta = %.2f, PVE = %.2f' %(unrotated_subspace['theta'],unrotated_subspace['pca'].explained_variance_ratio_.sum()))
    
    # plot rotated
    ax3 = plt.subplot(1,3,3, projection='3d')
    ax3.set_title('Rotated,Theta = %.2f, PVE = %.2f' %(rotated_subspace['theta'],rotated_subspace['pca'].explained_variance_ratio_.sum()))
    plot_geometry(ax3,rotated_subspace['3Dcoords'],rotated_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels=rotated_subspace['binned_data_subspace_order'])
    
    plt.tight_layout()
    
    
def get_AI_multiple_dims(precue_subspace,unrotated_subspace,rotated_subspace):
    precue_AI, unrotated_AI, rotated_AI = [], [], []
    for dim in range(2,6):
        precue_AI.append(get_simple_AI(precue_subspace['binned_data'][:4,:],
                                       precue_subspace['binned_data'][4:,:],dim))
        unrotated_AI.append(get_simple_AI(unrotated_subspace['binned_data'][:4,:],
                                       unrotated_subspace['binned_data'][4:,:],dim))
        rotated_AI.append(get_simple_AI(rotated_subspace['binned_data'][:4,:],
                                       rotated_subspace['binned_data'][4:,:],dim))
    
    

def get_cued_across_delay_subspaces_indivModels(constants,trial_type='valid'):
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/' 
    
    theta = np.empty((constants.PARAMS['n_models'],2))
    psi = np.empty((constants.PARAMS['n_models'],2))
    
    for model in range(constants.PARAMS['n_models']):
        # load data
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        # 'cued-up'
        delay2_up_trials = torch.cat((data_binned_cued['delay2'][:n_colours,:],
                                      data_binned_uncued['delay2'][:n_colours,:]))
        # 'cued-down'
        delay2_down_trials = torch.cat((data_binned_cued['delay2'][n_colours:,:],
                                        data_binned_uncued['delay2'][n_colours:,:]))
        
        
        delay2_up_trials_subspace = \
            run_pca_pipeline(constants,
                             delay2_up_trials,
                             ['cued_up','uncued_down'])
        delay2_down_trials_subspace = \
            run_pca_pipeline(constants, 
                             delay2_down_trials, 
                             ['cued_down','uncued_up'])
        
        trial_type_subspaces = {}
        trial_type_subspaces['cued_up'] = delay2_up_trials_subspace
        trial_type_subspaces['cued_down'] = delay2_down_trials_subspace
        
        # save angles
        theta[model,0] = delay2_up_trials_subspace['theta']
        theta[model,1] = delay2_down_trials_subspace['theta']
        
        psi[model,0] = delay2_up_trials_subspace['psi']
        psi[model,1] = delay2_down_trials_subspace['psi']

        pickle.dump(trial_type_subspaces,
                    open(load_path+'trial_type_subspaces_model'+str(model)+'.pckl','wb'))
        
    pickle.dump(theta,
                    open(load_path+'cued_vs_uncued_theta.pckl','wb'))
    pickle.dump(psi,
                    open(load_path+'cued_vs_uncued_psi.pckl','wb'))
    
    

def quick_plotter(model,unrotated_subspace,rotated_subspace):
    n_subplots = 2
    fsize = (12,5)
    
    
    plt.figure(figsize=fsize,num=('Model '+str(model)))
    ax = plt.subplot(1,n_subplots,1, projection='3d')
    plot_geometry(ax, 
                  unrotated_subspace['3Dcoords'],
                  unrotated_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  legend_on=False)
    plot_subspace(ax,
                  unrotated_subspace['3Dcoords'][:n_colours,:],
                  unrotated_subspace['plane1'].components_,fc='k',a=0.2)
    plot_subspace(ax,
                  unrotated_subspace['3Dcoords'][n_colours:,:],
                  unrotated_subspace['plane2'].components_,fc='k',a=0.2)
   
    ax.set_title('Unrotated angle: %.1f' %unrotated_subspace['theta'])
    helpers.equal_axes(ax)
    
    ax2 = plt.subplot(1,n_subplots,2, projection='3d')
    plot_geometry(ax2, 
                  rotated_subspace['3Dcoords'], 
                  rotated_subspace['pca'], 
                  constants.PLOT_PARAMS['4_colours'])
    ax2.set_title('Rotated angle: %.1f' %rotated_subspace['theta'])
    helpers.equal_axes(ax2)
    
    return ax,ax2





def get_AI_cued_across_delays(params,cv=2):
    '''
    For each location, get the AI between its subspaces from the 
    pre- and post-cue delay intervals. Do it in 2-fold cross-validation to 
    determine if both or only one of the pre-cue subspaces are are rotated post-cue
    to form a parallel plane geometry.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    AI_tbl : array
        AI values for the 'unrotated' and 'rotated' planes, averaged across cv 
        folds. Format: (n_dims,(unrotated,rotated),model)
    same_ixs : array
        Indexes of the unrotated plane for each model.
    trial_ixs : dict
        Train and test trial indexes for the cross-validation folds.

    '''
    load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    max_dim = 3
    n_dims = max_dim-1
    
    AI_table_ad_train = np.zeros((n_dims,2,params['n_models'],cv))
    AI_table_ad_test = np.zeros((n_dims,2,params['n_models'],cv))
    
    theta_ad_train = np.zeros((2,params['n_models'],cv))
    theta_ad_test = np.zeros((2,params['n_models'],cv))
    
    psi_ad_train = np.zeros((2,params['n_models'],cv))
    psi_ad_test = np.zeros((2,params['n_models'],cv))
    
    same_ixs_AI = np.zeros((2,params['n_models'],2))
    same_ixs_theta = np.zeros((params['n_models'],2))
    
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1
    
    trial_ixs = {'train':{},'test':{}}
    
    PVE = np.zeros((2,2,params['n_models'])) # plane1/2, unrot/rot
    
    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        train,test = get_trial_ixs(params,cv=2)
        trial_ixs['train'][model_number],trial_ixs['test'][model_number] = \
            train, test
        
        
        loc_split = params['B']
        for i in range(1):
            # bin the train and test datasets intro colour bins
            # data_train = helpers.bin_data(eval_data['data'][train[i],:,:],params)
            # data_test = helpers.bin_data(eval_data['data'][test[i],:,:],params)
            
            delay1_train = data_train[:,d1_ix,:]
            delay1_test = data_test[:,d1_ix,:]
            
            delay2_train = data_train[:,d2_ix,:]
            delay2_test = data_test[:,d2_ix,:]
            
            # # calculate the AI on train data 
            # for dim in range(2,max_dim+1):
            #     AI_table_ad_train[dim-2,0,model,i]=get_simple_AI(delay1_train[:loc_split,:],
            #                                                   delay2_train[:loc_split,:],max_dim) #loc1
            #     AI_table_ad_train[dim-2,1,model,i]=get_simple_AI(delay1_train[loc_split:,:],
            #                                                   delay2_train[loc_split:,:],max_dim) #loc2
            
            # # find the plane that stays the same
            # same_ixs_AI[:,model,i] = np.argmax(AI_table_ad_train[:,:,model,i],1)
            
            
            pca,_ = get_3D_coords(delay1_train[:loc_split,:])
            PVE[0,int(same_ixs_AI[0,model,i]),model] = pca.explained_variance_ratio_[:2].sum()
            
            pca,_ = get_3D_coords(delay2_train[:loc_split,:])
            PVE[1,int(same_ixs_AI[0,model,i]),model] = pca.explained_variance_ratio_[:2].sum()
            
            pca,_ = get_3D_coords(delay1_train[loc_split:,:])
            PVE[0,int(np.abs(1-same_ixs_AI[0,model,i])),model] = pca.explained_variance_ratio_[:2].sum()
            
            pca,_ = get_3D_coords(delay2_train[loc_split:,:])
            PVE[1,int(np.abs(1-same_ixs_AI[0,model,i])),model] = pca.explained_variance_ratio_[:2].sum()
            


#%%

from subspace_alignment_index import get_trial_ixs





#%%

def plot_all_subspaces():
    plt.figure()
    theta_pre, _ = get_angle_and_phase_between_planes_corrected(coords_3D_test[[0,1,2,3,8,9,10,11],:],
                                                unrot_plane1.components_,
                                                rot_plane1.components_)
    theta_post, _ = get_angle_and_phase_between_planes_corrected(coords_3D_test[[4,5,6,7,12,13,14,15],:],
                                                unrot_plane2.components_,
                                                rot_plane2.components_)
    
    # unrotated pre vs post
    ax = plt.subplot(221,projection='3d')
    plot_geometry(ax,coords_3D_test[:8,:],pca_test,
                  constants.PLOT_PARAMS['4_colours'],custom_labels=['unrot_pre','unrot_post'])
    # ax.set_title('Theta %.2f' %theta_unrotated_mean[model])
    ax.set_title('Theta %.2f' %theta_ad_test[0,model,i])
    # rotated pre vs post
    ax2 = plt.subplot(222,projection='3d')
    plot_geometry(ax2,coords_3D_test[8:,:],pca_test,
                  constants.PLOT_PARAMS['4_colours'],custom_labels=['rot_pre','rot_post'])
    # ax2.set_title('Theta %.2f' %theta_rotated_mean[model])
    ax2.set_title('Theta %.2f' %theta_ad_test[1,model,i])
    # pre
    ax3 = plt.subplot(223,projection='3d')
    plot_geometry(ax3,coords_3D_test[[0,1,2,3,8,9,10,11],:],pca_test,
                  constants.PLOT_PARAMS['4_colours'],custom_labels=['unrot_pre','rot_pre'])
    ax3.set_title('Theta %.2f' %theta_pre)
    
    ax4 = plt.subplot(224,projection='3d')
    plot_geometry(ax4,coords_3D_test[[4,5,6,7,12,13,14,15],:],pca_test,
                  constants.PLOT_PARAMS['4_colours'],custom_labels=['unrot_post','rot_post'])
    ax4.set_title('Theta %.2f' %theta_post)


#%%
def run_pca_pipeline(constants,data,subspace_order):
    
    n_colours = constants.PARAMS['B']
    
    # do PCA 1
    pca, coords_3D = get_3D_coords(data)
    
    # do PCA 2
    plane1 = get_best_fit_plane(coords_3D[:n_colours,:])
    plane2 = get_best_fit_plane(coords_3D[n_colours:8,:])
    
    # get angle between planes and phase alignment
    # theta = get_angle_between_planes_corrected(coords_3D,
    #                                            plane1.components_,
    #                                            plane2.components_)
    
    theta, pa = get_angle_and_phase_between_planes_corrected(coords_3D[:8,:],
                                                plane1.components_,
                                                plane2.components_)
    
    
    
    
    # make data structure
    subspace = {}
    subspace['binned_data'] = data
    subspace['binned_data_subspace_order'] = subspace_order
    subspace['pca'] = pca
    subspace['3Dcoords'] = coords_3D
    subspace['plane1'] = plane1
    subspace['plane2'] = plane2
    subspace['theta'] = theta
    subspace['psi'] = pa
    
    return subspace


def get_error_angle_estimates(cv=2):
    
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pre_post_angles = pickle.load(open(load_path+'all_theta.pckl','rb'))
    
    pre_post_estimated = np.zeros((params['n_models'],2,cv))
    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        train,test = get_trial_ixs(params,cv=2)
        trial_ixs['train'][model_number],trial_ixs['test'][model_number] = \
            train, test
        
        
        loc_split = params['B']
        for i in range(2):
            # bin the train and test datasets into colour bins
            data_train = helpers.bin_data(params, eval_data['data'][train[i],:,:])
            data_test = helpers.bin_data(params, eval_data['data'][test[i],:,:])
            
            delay1_train = data_train[:,d1_ix,:]
            delay1_test = data_test[:,d1_ix,:] - data_test[:,d1_ix,:].mean()
            
            delay2_train = data_train[:,d2_ix,:]
            delay2_test = data_test[:,d2_ix,:] - data_test[:,d2_ix,:].mean()
            
    
    # loc1_subspace = run_pca_pipeline(constants,
    #                   torch.cat((delay1_train[:loc_split,:],
    #                                                   delay2_train[:loc_split,:])),
    #                   ['pre_delay','post_delay']) #loc1
    # # loc 2 cued
    # loc2_subspace = run_pca_pipeline(constants,
    #                   torch.cat((delay1_train[loc_split:,:],
    #                                                   delay2_train[loc_split:,:])),
    #                   ['pre_delay','post_delay']) #loc2
    
    # same_ixs_theta[model,i] = np.argmax([np.abs(np.cos(np.radians(loc1_subspace['theta']))),
    #                                   np.abs(np.cos(np.radians(loc2_subspace['theta'])))])
    
            all_data_pre = torch.cat((delay1_test[stay_plane_ix,:],
                                            delay1_test[switch_plane_ix,:],
                                            delay2_test[stay_plane_ix,:],
                                            delay2_test[switch_plane_ix,:]))
           
            all_data_post = torch.cat((delay2_test[stay_plane_ix,:],
                                            delay2_test[switch_plane_ix,:],
                                            delay1_test[stay_plane_ix,:],
                                            delay1_test[switch_plane_ix,:],))
            
            subspace_pre = run_pca_pipeline(constants,all_data_pre,['c1','c2'])
           
            subspace_post = run_pca_pipeline(constants,all_data_post,['c1','c2'])


            pre_post_estimated[model,0,i] = subspace_pre['theta']
            pre_post_estimated[model,1,i] = subspace_post['theta']
    
    pre_estimated_mean = np.array([pre_post_estimated[m,0,:].mean(-1) if np.sign(pre_post_estimated[m,0,:]).sum()!=0 else np.abs(pre_post_estimated[m,0,:]).mean(-1) for m in range(params['n_models'])])
    post_estimated_mean = np.array([pre_post_estimated[m,1,:].mean(-1) if np.sign(pre_post_estimated[m,1,:]).sum()!=0 else np.abs(pre_post_estimated[m,1,:]).mean(-1) for m in range(params['n_models'])])
    
    
    err_pre = np.abs(pre_post_angles[:,0])-np.abs(pre_estimated_mean)
    err_post = np.abs(pre_post_angles[:,1])-np.abs(post_estimated_mean)
    
plot_plane_angles_single(constants,np.radians(err_pre),'cu',r=None)
plt.title('Error in pre-cue angle estimate')
plot_plane_angles_single(constants,np.radians(err_post),'cu',r=None)
plt.title('Error in post-cue angle estimate')
