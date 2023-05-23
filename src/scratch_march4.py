#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:06:20 2023

@author: emilia
"""


# load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/' 
load_path = constants.PARAMS['FULL_PATH']+'pca_data/invalid_trials/' 
d3_ix = constants.PARAMS['trial_timepoints']['delay3_end']-1
for model in range(constants.PARAMS['n_models']):
    path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
    path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

    data_binned_cued = pickle.load(open(path_cued,'rb'))
    data_binned_uncued = pickle.load(open(path_uncued,'rb'))
    
    data_binned_cued['delay3'] = data_binned_cued['data'][:,d3_ix,:]
    data_binned_uncued['delay3'] = data_binned_uncued['data'][:,d3_ix,:]
    
    pickle.dump(data_binned_cued,open(path_cued,'wb'))
    pickle.dump(data_binned_uncued,open(path_uncued,'wb'))
    
    
    
    
    
    
def get_decoding_within_plane(constants,which_delay):
    '''
    Train and test LDA binary classifiers to discriminate between pairs of 
     colour labels in the pre or post-cue delay (at the last time-point). For 
     the post-cue delay, decoding is done for the uncued colours only.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    which_delay : str
        Name of the memory delay to extract the data from. Pass 'precue' or 
        'postcue'.

    Returns
    -------
    model_scores : array
        Average test decoding acuracy scores for all models.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    model_scores = np.empty((constants.PARAMS['n_models'],))
    
    if which_delay == 'precue':
        delay_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
        fname_str = 'precue'
    else:
        # post-cue delay (uncued colours)
        delay_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
        fname_str = 'uncued_postcue'
        
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        print('Model '+ model_number +'/' + str(constants.PARAMS['n_models']))
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        n_trials = eval_data['data'].shape[0]
        
        
        
        cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                             data_binned_uncued['delay1'][:n_colours,:],
                               data_binned_cued['delay2'][:n_colours,:],
                               data_binned_uncued['delay2'][:n_colours,:],
                               data_binned_cued['delay3'][:n_colours,:],
                               data_binned_uncued['delay3'][:n_colours,:]))
        
        
        # get the uncued colour labels
        labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                                  eval_data["labels"]["c1"][n_trials//2:]))
        # bin the labels into B colour bins
        labels_uncued_binned = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
        # split the labels according to the cued location
        loc1_labels = labels_uncued_binned[:n_trials//2]
        loc2_labels = labels_uncued_binned[n_trials//2:]
        
        delayloc1 = eval_data['data'][:n_trials//2,delay_ix,:]
        delayloc2 = eval_data['data'][n_trials//2:,delay_ix,:]
        
        # train and test LDA classifiers
        scores_loc1 = lda(delayloc1,loc1_labels)
        scores_loc2 = lda(delayloc2,loc2_labels)
        
        # save LDA model test scores
        model_scores[model] = np.stack((scores_loc1,scores_loc2)).mean()
    # save to file 
    pickle.dump(model_scores,open(load_path+'/decoding_acc_'+fname_str+'_delay.pckl','wb'))
    return model_scores



#%%

for model in range(constants.PARAMS['n_models']):
    path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
    path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

    data_binned_cued = pickle.load(open(path_cued,'rb'))
    data_binned_uncued = pickle.load(open(path_uncued,'rb'))
    
    if constants.PARAMS['experiment_number'] == 3:
        # cued-up trials
        cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                             data_binned_uncued['delay1'][:n_colours,:],
                               data_binned_cued['delay2'][:n_colours,:],
                               data_binned_uncued['delay2'][:n_colours,:],
                               data_binned_cued['delay3'][:n_colours,:],
                               data_binned_uncued['delay3'][:n_colours,:]))
        #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
    else:
        # cued-up trials
        cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                             data_binned_uncued['delay1'][:n_colours,:],
                               data_binned_cued['delay2'][:n_colours,:],
                               data_binned_uncued['delay2'][:n_colours,:]))
        #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
    
    # l = [data_binned_cued['delay1'][:n_colours,:],
    #                      data_binned_uncued['delay1'][:n_colours,:],
    #                        data_binned_cued['delay2'][:n_colours,:],
    #                        data_binned_uncued['delay2'][:n_colours,:]]
    # get the common subspace for cued-up trials
    n_timepoints=2
    cued_up_3Dcoords = np.zeros((n_colours*2*n_timepoints,3))
    cued_up_pca = []
    for i in range(4):
        cued_up_pca.append([])
        ix = np.arange(i*4,(i+1)*4)
        cued_up_pca[i], cued_up_3Dcoords[ix,:] = get_3D_coords(cued_up[ix,:])
        
        
    # loc1pre, loc1pre_3Dcoords = get_3D_coords(data_binned_cued['delay1'][:n_colours,:])
    # loc2pre, loc2pre_3Dcoords = get_3D_coords(data_binned_uncued['delay1'][:n_colours,:])
    # loc1cued,loc1cued_3Dcoords = get_3D_coords(data_binned_cued['delay2'][:n_colours,:])
    # loc2uncued, loc2uncued_3Dcoords = get_3D_coords(data_binned_uncued['delay2'][:n_colours,:])
    
    # CDI[model,0,0,0] = quadrilatArea(loc1pre_3Dcoords)
    # CDI[model,0,0,1] = quadrilatArea(loc2pre_3Dcoords)
    # #post-cue
    # CDI[model,0,1,0] = quadrilatArea(loc1cued_3Dcoords)
    # CDI[model,0,1,1] = quadrilatArea(loc2uncued_3Dcoords)
    
    
    
    # cued_up_pca, cued_up_3Dcoords = get_3D_coords(cued_up)
    # PVEs[model,0] = cued_up_pca.explained_variance_ratio_.sum()
    # pdb.set_trace()
    #pre-cue
    # CDI[model,0,0,0] = ConvexHull(cued_up_3Dcoords[:n_colours,:]).area
    # CDI[model,0,0,1] = ConvexHull(cued_up_3Dcoords[n_colours:n_colours*2,:]).area
    # #post-cue
    # CDI[model,0,1,0] = ConvexHull(cued_up_3Dcoords[n_colours*2:n_colours*3,:]).area
    # CDI[model,0,1,1] = ConvexHull(cued_up_3Dcoords[n_colours*3:,:]).area
    
    #pre-cue
    CDI[model,0,0,0] = quadrilatArea(cued_up_3Dcoords[:n_colours,:])
    CDI[model,0,0,1] = quadrilatArea(cued_up_3Dcoords[n_colours:n_colours*2,:])
    #post-cue
    CDI[model,0,1,0] = quadrilatArea(cued_up_3Dcoords[n_colours*2:n_colours*3,:])
    CDI[model,0,1,1] = quadrilatArea(cued_up_3Dcoords[n_colours*3:n_colours*4,:])
    

#%%

def get_CDI(constants,trials='valid'):
    n_colours = constants.PARAMS['B']
    if trials == 'valid':
        load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/' 
    else:
        load_path = constants.PARAMS['FULL_PATH']+'pca_data/invalid_trials/' 
    
    if constants.PARAMS['experiment_number'] == 3:
        n_timepoints=3
         
    else:
        n_timepoints=2
    # model, trial type (cued location), pre/post-cue, plane1/plane2
    CDI = np.empty((constants.PARAMS['n_models'],2,n_timepoints,2))  
    PVEs = np.empty((constants.PARAMS['n_models'],2)) # model, trial type
    #[model, cued location, pre-post, cued/uncued]
    for model in range(constants.PARAMS['n_models']):
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        if constants.PARAMS['experiment_number'] == 3:
            # cued-up trials
            cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                                 data_binned_uncued['delay1'][:n_colours,:],
                                   data_binned_cued['delay2'][:n_colours,:],
                                   data_binned_uncued['delay2'][:n_colours,:],
                                   data_binned_cued['delay3'][:n_colours,:],
                                   data_binned_uncued['delay3'][:n_colours,:]))
            #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
        else:
            # cued-up trials
            cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                                 data_binned_uncued['delay1'][:n_colours,:],
                                   data_binned_cued['delay2'][:n_colours,:],
                                   data_binned_uncued['delay2'][:n_colours,:]))
            #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
        
        
        # # get the common subspace for cued-up trials
        # cued_up_pca, cued_up_3Dcoords = get_3D_coords(cued_up)
        # PVEs[model,0] = cued_up_pca.explained_variance_ratio_.sum()
        
        cued_up_3Dcoords = np.zeros((n_colours*2*n_timepoints,3))
        cued_up_pca = []
        for i in range(2*n_timepoints):
            cued_up_pca.append([])
            ix = np.arange(i*4,(i+1)*4)
            cued_up_pca[i], cued_up_3Dcoords[ix,:] = get_3D_coords(cued_up[ix,:])
        
        
        #pre-cue
        CDI[model,0,0,0] = quadrilatArea(cued_up_3Dcoords[:n_colours,:])
        CDI[model,0,0,1] = quadrilatArea(cued_up_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,0,1,0] = quadrilatArea(cued_up_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,0,1,1] = quadrilatArea(cued_up_3Dcoords[n_colours*3:n_colours*4,:])
        
        if constants.PARAMS['experiment_number'] == 3:
            #post-probe
            CDI[model,0,2,0] = quadrilatArea(cued_up_3Dcoords[n_colours*4:n_colours*5,:])
            CDI[model,0,2,1] = quadrilatArea(cued_up_3Dcoords[n_colours*5:,:])
        
        
        if constants.PARAMS['experiment_number'] == 3:
            # cued-down trials
            cued_down = torch.cat((data_binned_cued['delay1'][n_colours:,:],
                                   data_binned_uncued['delay1'][n_colours:,:],
                                   data_binned_cued['delay2'][n_colours:,:],
                                   data_binned_uncued['delay2'][n_colours:,:],
                                   data_binned_cued['delay3'][n_colours:,:],
                                   data_binned_uncued['delay3'][n_colours:,:]))
        else:
            # cued-down trials
            cued_down = torch.cat((data_binned_cued['delay1'][n_colours:,:],
                                   data_binned_uncued['delay1'][n_colours:,:],
                                   data_binned_cued['delay2'][n_colours:,:],
                                   data_binned_uncued['delay2'][n_colours:,:]))    
        # pre-cue:down; pre-cue:up;post-cue:down;post-cue:up
        # get the common subspace for cued-down trials
        # cued_down_pca, cued_down_3Dcoords = get_3D_coords(cued_down)
        # PVEs[model,1] = cued_down_pca.explained_variance_ratio_.sum()
        
        cued_down_3Dcoords = np.zeros((n_colours*2*n_timepoints,3))
        cued_down_pca = []
        for i in range(2*n_timepoints):
            cued_down_pca.append([])
            ix = np.arange(i*4,(i+1)*4)
            cued_down_pca[i], cued_down_3Dcoords[ix,:] = get_3D_coords(cued_down[ix,:])
            
        #pre-cue
        CDI[model,1,0,0] = quadrilatArea(cued_down_3Dcoords[:n_colours,:])
        CDI[model,1,0,1] = quadrilatArea(cued_down_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,1,1,0] = quadrilatArea(cued_down_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,1,1,1] = quadrilatArea(cued_down_3Dcoords[n_colours*3:n_colours*4,:])
        
        if constants.PARAMS['experiment_number'] == 3:
            #post-probe
            CDI[model,1,2,0] = quadrilatArea(cued_down_3Dcoords[n_colours*4:n_colours*5,:])
            CDI[model,1,2,1] = quadrilatArea(cued_down_3Dcoords[n_colours*5:,:])
    # average across the trial types
    CDI_av = CDI.mean(1)
    # for pre-cue, average the cued and uncued
    if constants.PARAMS['experiment_number'] == 3:
        CDI_av = np.concatenate((np.expand_dims(CDI_av[:,0,:].mean(-1),-1),CDI_av[:,1,:],CDI_av[:,2,:]),1)
    else:
        CDI_av = np.concatenate((np.expand_dims(CDI_av[:,0,:].mean(-1),-1),CDI_av[:,1,:]),1)
        CDI_av_df = pd.DataFrame(CDI_av,columns=['pre-cue','post_cued','post_uncued','probed','unprobed'])
    
    pickle.dump(CDI_av,open(load_path+'/CDI_'+trials+'_trials'+'.pckl','wb'))
    # # save structure
    CDI_av_df.to_csv(load_path+'/CDI_'+trials+'_trials'+'.csv')
    mean_PVE = PVEs.mean()*100
    sem_PVE = np.std(PVEs.mean(-1))*100 / np.sqrt(constants.PARAMS['n_models'])
    print('%% variance explained by 3D subspaces mean = %.2f, sem = %.3f' %(mean_PVE,sem_PVE))
    return CDI_av


ax = plt.subplot(111, projection='3d',custom_labels=[])
def quick_plot(ax,coords_3d,pca):
    plot_geometry(ax,
                  coords_3d,
                  pca, 
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels = custom_labels)
    plot_subspace(ax,coords_3d[:n_colours,:],data['plane1'].components_,fc='k',a=0.2)
    plot_subspace(ax,data['3Dcoords'][n_colours:,:],data['plane2'].components_,fc='k',a=0.2)
    ax.set_title(plot_title + ', ' + r'$\theta$' + ' = %.1f' %data['theta']+'°')
    helpers.equal_axes(ax)




#%% project both vaid and invaid trials into the same subspace




def check_pca_data(constants,model=0):
    cued_up = []
    cued_down = []
    for trials in ['valid','invalid']:
        load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trials+'_trials/' 
        
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'
        
        if trials == 'valid':
            data_binned_cued = pickle.load(open(path_cued,'rb'))
            data_binned_uncued = pickle.load(open(path_uncued,'rb'))
            eval_data = pickle.load(open(load_path+'eval_data_model'+str(model)+'.pckl','rb'))
            eval_data_uncued = pickle.load(open(load_path+'eval_data_uncued_model'+str(model)+'.pckl','rb'))
        else:
            data_binned_cued_invalid = pickle.load(open(path_cued,'rb'))
            data_binned_uncued_invalid = pickle.load(open(path_uncued,'rb'))
            eval_data_invalid = pickle.load(open(load_path+'eval_data_model'+str(model)+'.pckl','rb'))
            eval_data_uncued_invalid = pickle.load(open(load_path+'eval_data_uncued_model'+str(model)+'.pckl','rb'))
    
    
    d1_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
    d2_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    d3_ix = constants.PARAMS['trial_timepoints']['delay3_end']-1
    # eval data
    # here the order of trials should be the same, the network does not yet know
    # which colour is cued/un- or probed/-un plus the data was not shuffled
    torch.all(eval_data['data'][:,d1_ix,:]==eval_data_uncued_invalid['data'][:,d1_ix,:])
    # this data was shuffled, but should be in the same order
    torch.all(eval_data_uncued['data'][:,d1_ix,:]==eval_data_invalid['data'][:,d1_ix,:])
    
    # test counterfactuals - this will tell us if some of the data was truly shuffled
    not torch.all(eval_data['data'][:,d1_ix,:]==eval_data_invalid['data'][:,d1_ix,:])
    # this data was shuffled, but should be in the same order
    not torch.all(eval_data_uncued['data'][:,d1_ix,:]==eval_data_uncued_invalid['data'][:,d1_ix,:])
    
    # this compares shuffled vs reshuffled data (so different half of the data)
    # should be equal - should be the same for d1 because no cue has been presented
    n_trials = eval_data['data'].shape[0]
    torch.all(eval_data['data'][:n_trials//2,d1_ix,:]==eval_data_invalid['data'][n_trials//2:,d1_ix,:])
    torch.all(eval_data['data'][n_trials//2:,d1_ix,:]==eval_data_invalid['data'][:n_trials//2,d1_ix,:])
    torch.all(eval_data_uncued['data'][:n_trials//2,d1_ix,:]==eval_data_uncued_invalid['data'][n_trials//2:,d1_ix,:])
    torch.all(eval_data_uncued['data'][n_trials//2:,d1_ix,:]==eval_data_uncued_invalid['data'][:n_trials//2,d1_ix,:])
    torch.all(eval_data['data'][:n_trials//2,d1_ix,:]==eval_data_uncued['data'][n_trials//2:,d1_ix,:])
    torch.all(eval_data['data'][n_trials//2:,d1_ix,:]==eval_data_uncued['data'][:n_trials//2,d1_ix,:])
    torch.all(eval_data_invalid['data'][:n_trials//2,d1_ix,:]==eval_data_uncued_invalid['data'][n_trials//2:,d1_ix,:])
    torch.all(eval_data_invalid['data'][n_trials//2:,d1_ix,:]==eval_data_uncued_invalid['data'][:n_trials//2,d1_ix,:])
    
    # now test counterfactuals
    not torch.all(eval_data['data'][:n_trials//2,d1_ix,:]==eval_data_invalid['data'][:n_trials//2,d1_ix,:])
    not torch.all(eval_data['data'][n_trials//2:,d1_ix,:]==eval_data_invalid['data'][n_trials//2:,d1_ix,:])
    not torch.all(eval_data_uncued['data'][:n_trials//2,d1_ix,:]==eval_data_uncued_invalid['data'][:n_trials//2,d1_ix,:])
    not torch.all(eval_data_uncued['data'][n_trials//2:,d1_ix,:]==eval_data_uncued_invalid['data'][n_trials//2,d1_ix,:])
    not torch.all(eval_data['data'][:n_trials//2,d1_ix,:]==eval_data_uncued['data'][:n_trials//2,d1_ix,:])
    not torch.all(eval_data['data'][n_trials//2:,d1_ix,:]==eval_data_uncued['data'][n_trials//2:,d1_ix,:])
    not torch.all(eval_data_invalid['data'][:n_trials//2,d1_ix,:]==eval_data_uncued_invalid['data'][:n_trials//2,d1_ix,:])
    not torch.all(eval_data_invalid['data'][n_trials//2:,d1_ix,:]==eval_data_uncued_invalid['data'][n_trials//2:,d1_ix,:])
    
    # so eval data dchecks out    
    
    # now test pca_data
    # cued-valid and uncued-invalid should be exactly equal in pre-cue delay
    torch.all(data_binned_cued['delay1']==data_binned_uncued_invalid['delay1'])
    torch.all(data_binned_uncued['delay1']==data_binned_cued_invalid['delay1'])
    # test counterfactuals 
    not torch.all(data_binned_cued['delay1']==data_binned_cued_invalid['delay1'])
    not torch.all(data_binned_uncued['delay1']==data_binned_uncued_invalid['delay1'])
        
    
    # test reshuffled halfs
    torch.all(data_binned_cued['delay1'][:4,:]==data_binned_uncued['delay1'][4:,:])
    torch.all(data_binned_cued['delay1'][4:,:]==data_binned_uncued['delay1'][:4,:])
    
    torch.all(data_binned_cued['delay1'][:4,:]==data_binned_cued_invalid['delay1'][4:,:])
    torch.all(data_binned_cued['delay1'][4:,:]==data_binned_cued_invalid['delay1'][:4,:])
    torch.all(data_binned_uncued['delay1'][:4,:]==data_binned_cued_invalid['delay1'][4:,:])
    
    # are the cued reps equal for the same cue between valid and invalid trials?
    torch.all(data_binned_cued['delay2'][:4,:]==data_binned_cued_invalid['delay2'][4:,:]) #loc1 cued
    torch.all(data_binned_cued['delay2'][4:,:]==data_binned_cued_invalid['delay2'][:4,:]) # loc2 cued
    
    # what about uncued
    torch.all(data_binned_uncued['delay2'][:4,:]==data_binned_uncued_invalid['delay2'][4:,:]) #loc2 uncued
    torch.all(data_binned_uncued['delay2'][4:,:]==data_binned_uncued_invalid['delay2'][:4,:]) # loc1 uncued
    
    # are the probed reps different    
    
    
def plot_all_labels(eval_data,eval_data_uncued,eval_data_invalid,eval_data_uncued_invalid):
    all_data = [eval_data,eval_data_uncued,eval_data_invalid,eval_data_uncued_invalid]
    all_data_names = ['eval_data_cued','eval_data_uncued','eval_data_cued_invalid','eval_data_uncued_invalid']
    
    for data_name,data in zip(all_data_names,all_data):
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.repeat(data['labels']['c1'][:,None],1000,1))
        plt.title('c1')
        
        plt.subplot(122)
        plt.imshow(np.repeat(data['labels']['c2'][:,None],1000,1))
        plt.title('c2')
        
        plt.suptitle(data_name)
        
        
def test_binning():
    for i in range(8):
        print(torch.all(trial_data[i,:,:,:]==eval_data['data'][n_samples*i:n_samples*(i+1),:,:]))
        
    trial_data2 = bin_data(eval_data['data'],params)
    for i in range(8):
        print(torch.all(trial_data2[i,:,:,:]==eval_data['data'][n_samples*i:n_samples*(i+1),:,:]))
    
    trial_data = trial_data.mean(1) #(M,seq_len,n_rec)
    
    trial_data2 = helpers.bin_data(eval_data['data'],params)
    
    
    binned_cued_valid = bin_data(eval_data['data'],constants.PARAMS)
    binned_uncued_valid = bin_data(eval_data_uncued['data'],constants.PARAMS)
    
    
    
    M = constants.PARAMS['M']
    
    # bin into M colour bins
    n_samples = eval_data['data'].shape[0]//M
    # if len(data.shape) == 2:
    #     # matrix with single timepoint
    #     data_binned = data.unsqueeze(0).reshape((M,n_samples,params['n_rec']))
    # elif len(data.shape) == 3:
        # matrix with multiple timepoints
    n_timepoints = eval_data['data'].shape[1]
    
    binned_cued_valid =  eval_data['data'].unsqueeze(0).reshape((M,n_samples,
                                n_timepoints,constants.PARAMS['n_rec']))
        
    binned_uncued_valid = eval_data_uncued['data'].unsqueeze(0).reshape((M,n_samples,
                                n_timepoints,constants.PARAMS['n_rec']))
    # data_binned = data_binned.mean(1)
    binned_cued_valid_b = binned_cued_valid.mean(1)
    binned_uncued_valid_b = binned_uncued_valid.mean(1)
    
    binned_cued_valid_mean_np = binned_cued_valid.numpy().mean(1)
    binned_uncued_valid_mean_np = binned_uncued_valid.numpy().mean(1)
    
    binned_cued_valid_mean_torch = binned_cued_valid.mean(1)
    binned_uncued_valid_mean_torch = binned_uncued_valid.mean(1)
    
    
    
    binned_cued_valid_bb = torch.zeros((M,n_timepoints,constants.PARAMS['n_rec']))
    binned_uncued_valid_bb = torch.zeros((M,n_timepoints,constants.PARAMS['n_rec']))
    
    binned_cued_valid_mean_torch_looped = torch.zeros((M,n_timepoints,constants.PARAMS['n_rec']))
    
    for b in range(M//2):
        for t in range(n_timepoints):
            for n in range(constants.PARAMS['n_rec']):
                
                binned_cued_valid_bb[b,t,n] = binned_cued_valid[b,:,t,n].mean(0)
                binned_uncued_valid_bb[b+4,t,n] = binned_uncued_valid[b+4,:,t,n].mean(0)
                
    
    
    
    binned_cued_valid_bb=torch.stack(binned_cued_valid_bb)
    binned_uncued_valid_bb=torch.stack(binned_uncued_valid_bb)
    
    
    
a1 = np.random.randn(5,5,5,5)

mean_np = a1.mean(1)
mean_torch = torch.from_numpy(a1).mean(1)


#%%

def plot_CDI_single_model_unaveraged(CDI,model=0):
    time_labels = ['precue_','postcue_','postprobe_']
    loc_labels = ['L1','L2']
    
    labels = [tt + l for tt in time_labels for l in loc_labels]
    
    # plot CDI values on cued-up (L1) trials
    plt.figure()
    plt.plot(np.log(CDI[model,0,:,:].flatten()),'k-')
    plt.plot(np.arange(0,12,2),np.log(CDI[model,0,:,:].flatten())[0:12:2],'go',label='valid trials')
    plt.plot(np.arange(1,13,2),np.log(CDI[model,0,:,:].flatten())[1:13:2],'ro',label='invalid trials')
    plt.xticks(np.arange(0.5,12.5,2),labels=labels)#,rotation=45)
    plt.ylabel('log CDI')
    plt.title('Model {}, data from L1 cued trials'.format(model))
    plt.legend()
    
    # plot CDI values on cued-down (L2) trials
    plt.figure()
    plt.plot(np.log(CDI[model,1,:,:].flatten()),'k-')
    plt.plot(np.arange(0,12,2),np.log(CDI[model,1,:,:].flatten())[0:12:2],'go',label='valid trials')
    plt.plot(np.arange(1,13,2),np.log(CDI[model,1,:,:].flatten())[1:13:2],'ro',label='invalid trials')
    plt.xticks(np.arange(0.5,12.5,2),labels=labels)#,rotation=45)
    plt.ylabel('log CDI')
    plt.title('Model {}, data from L2 cued trials'.format(model))
    plt.legend()
    return





def plot_CDI_old(constants,CDI,logTransform=True,save_fig=True,trials='valid'):
    
    if logTransform:
        CDI = np.log(CDI)
    pal = sns.color_palette("dark")
    
    if constants.PARAMS['experiment_number']==3:
        cols = [pal[9],pal[2],pal[6],pal[4]]
    else:
        cols = [pal[9],pal[2]]
    
    plt.figure(figsize=(6.65,5))
    ax = plt.subplot(111)
    
    ms = 16
    
    if trials == 'valid':
        cued_ix, uncued_ix = 1,2
        offsets = [-.125,.125]
    else:
        cued_ix, uncued_ix = 2,1
        offsets = [.125,-.125]
    
    for model in range(0,constants.PARAMS['n_models']):
        ax.plot([0,1-.125],CDI[model,[0,cued_ix]],'k-',alpha=.2)
        ax.plot([0,1+.125],CDI[model,[0,uncued_ix]],'k-',alpha=.2)
        
        if constants.PARAMS['experiment_number']==3:
            
            ax.plot([1+offsets[0],2-.125],CDI[model,[1,3]],'k-',alpha=.2)
            ax.plot([1+offsets[1],2+.125],CDI[model,[2,4]],'k-',alpha=.2)
        
        

        ax.plot(0,CDI[model,0],'o',c='k',markersize=ms) # pre-cue
        ax.plot(1-.125,CDI[model,cued_ix],'^',c=cols[0],markersize=ms) # cued
        ax.plot(1+.125,CDI[model,uncued_ix],'X',c=cols[1],markersize=ms) # uncued
        
        if constants.PARAMS['experiment_number']==3:
            ax.plot(2-.125,CDI[model,3],'s',c=cols[2],markersize=ms) # probed
            ax.plot(2+.125,CDI[model,4],'d',c=cols[3],markersize=ms) # unprobed

    # add means
    means = CDI.mean(0)
    ax.bar(0,means[0],facecolor='k',alpha=.2,width=.25)
    
        
    ax.bar(1-.125,means[cued_ix],facecolor=cols[0],alpha=.2,width=.25,label='cued')
    ax.bar(1+.125,means[uncued_ix],facecolor=cols[1],alpha=.2,width=.25,label='uncued')
    
    if constants.PARAMS['experiment_number']==3:
        ax.bar(2-.125,means[3],facecolor=cols[2],alpha=.2,width=.25,label='probed')
        ax.bar(2+.125,means[4],facecolor=cols[3],alpha=.2,width=.25,label='unprobed')
        ax.set_xlim([-0.25,2.375])
    else:
        ax.set_xlim([-0.25,1.375])

    
    # for expt 3
    # trials == 'valid'
    
    
    # x_vals = np.arange(2)
    # sem_cued = np.std(CDI.mean(1)[:,:,0],0) / np.sqrt(constants.PARAMS['n_models'])
    # sem_uncued = np.std(CDI.mean(1)[:,:,1],0) / np.sqrt(constants.PARAMS['n_models'])

    # ax.errorbar(x_vals,
    #             CDI.mean(1)[:,:,0].mean(0),
    #             fmt='.-',
    #             yerr = sem_cued,
    #             c = cols[0],
    #             label='cued')
    #             # ecolor=cols[0],
    #             # mfc=cols[0],
    #             # mec=cols[0])
    # ax.errorbar(x_vals,
    #             CDI.mean(1)[:,:,1].mean(0),
    #             fmt='.-',
    #             yerr = sem_uncued,
    #             c=cols[1],
    #             label = 'uncued')
                # mfc=cols[1],
                # mec=cols[1])
    if constants.PARAMS['experiment_number']==3:
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels=['pre-cue', 'post-cue','post-probe'])
    else:
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels=['pre-cue', 'post-cue'])
    # ax.set_xlim([-.25,1.25])
    if logTransform:
        ax.set_ylabel('log(CDI)')
    else:
        ax.set_ylabel('CDI')
    
    plt.legend()
    plt.tight_layout()
    
    if save_fig == True:
        plt.savefig(constants.PARAMS['FIG_PATH']+'CDI.png')
        
    return ax


def get_CDI_old(constants,trials='valid'):
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trials+'_trials/' 
    
    if constants.PARAMS['experiment_number'] == 3:
        CDI = np.empty((constants.PARAMS['n_models'],2,3,2))   
    else:
        CDI = np.empty((constants.PARAMS['n_models'],2,2,2))
        # model, trial type (cued location), pre/post-cue, plane1/plane2
    PVEs = np.empty((constants.PARAMS['n_models'],2)) # model, trial type
    #[model, cued location, pre-post, cued/uncued]
    for model in range(constants.PARAMS['n_models']):
        
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        
        if constants.PARAMS['experiment_number'] == 3:
            path_probed =  load_path + 'pca_data_probed_model' + str(model) + '.pckl'
            path_unprobed =  load_path + 'pca_data_unprobed_model' + str(model) + '.pckl'

            data_binned_probed = pickle.load(open(path_probed,'rb'))
            data_binned_unprobed = pickle.load(open(path_unprobed,'rb'))
            
            # cued-up trials
            cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                                 data_binned_uncued['delay1'][:n_colours,:],
                                   data_binned_cued['delay2'][:n_colours,:],
                                   data_binned_uncued['delay2'][:n_colours,:],
                                   data_binned_cued['delay3'][:n_colours,:],
                                   data_binned_uncued['delay3'][:n_colours,:]))
            #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
        else:
            # cued-up trials
            cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                                 data_binned_uncued['delay1'][:n_colours,:],
                                   data_binned_cued['delay2'][:n_colours,:],
                                   data_binned_uncued['delay2'][:n_colours,:]))
            #pre-cue:up;pre-cue:down;post-cue:up;post-cue:down
        
        
        # get the common subspace for cued-up trials
        cued_up_pca, cued_up_3Dcoords = get_3D_coords(cued_up)
        PVEs[model,0] = cued_up_pca.explained_variance_ratio_.sum()
        # pdb.set_trace()
        #pre-cue
        # CDI[model,0,0,0] = ConvexHull(cued_up_3Dcoords[:n_colours,:]).area
        # CDI[model,0,0,1] = ConvexHull(cued_up_3Dcoords[n_colours:n_colours*2,:]).area
        # #post-cue
        # CDI[model,0,1,0] = ConvexHull(cued_up_3Dcoords[n_colours*2:n_colours*3,:]).area
        # CDI[model,0,1,1] = ConvexHull(cued_up_3Dcoords[n_colours*3:,:]).area
        
        #pre-cue
        CDI[model,0,0,0] = quadrilatArea(cued_up_3Dcoords[:n_colours,:])
        CDI[model,0,0,1] = quadrilatArea(cued_up_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,0,1,0] = quadrilatArea(cued_up_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,0,1,1] = quadrilatArea(cued_up_3Dcoords[n_colours*3:n_colours*4,:])
        
        if constants.PARAMS['experiment_number'] == 3:
            #post-probe
            CDI[model,0,2,0] = quadrilatArea(cued_up_3Dcoords[n_colours*4:n_colours*5,:])
            CDI[model,0,2,1] = quadrilatArea(cued_up_3Dcoords[n_colours*5:,:])
        
        
        if constants.PARAMS['experiment_number'] == 3:
            # cued-down trials
            cued_down = torch.cat((data_binned_cued['delay1'][n_colours:,:],
                                   data_binned_uncued['delay1'][n_colours:,:],
                                   data_binned_cued['delay2'][n_colours:,:],
                                   data_binned_uncued['delay2'][n_colours:,:],
                                   data_binned_cued['delay3'][n_colours:,:],
                                   data_binned_uncued['delay3'][n_colours:,:]))
        else:
            # cued-down trials
            cued_down = torch.cat((data_binned_cued['delay1'][n_colours:,:],
                                   data_binned_uncued['delay1'][n_colours:,:],
                                   data_binned_cued['delay2'][n_colours:,:],
                                   data_binned_uncued['delay2'][n_colours:,:]))    
        # pre-cue:down; pre-cue:up;post-cue:down;post-cue:up
        # get the common subspace for cued-down trials
        cued_down_pca, cued_down_3Dcoords = get_3D_coords(cued_down)
        PVEs[model,1] = cued_down_pca.explained_variance_ratio_.sum()
        
        # #pre-cue
        # CDI[model,1,0,0] = ConvexHull(cued_down_3Dcoords[:n_colours,:]).area
        # CDI[model,1,0,1] = ConvexHull(cued_down_3Dcoords[n_colours:n_colours*2,:]).area
        # #post-cue
        # CDI[model,1,1,0] = ConvexHull(cued_down_3Dcoords[n_colours*2:n_colours*3,:]).area
        # CDI[model,1,1,1] = ConvexHull(cued_down_3Dcoords[n_colours*3:,:]).area
        
        #pre-cue
        CDI[model,1,0,0] = quadrilatArea(cued_down_3Dcoords[:n_colours,:])
        CDI[model,1,0,1] = quadrilatArea(cued_down_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,1,1,0] = quadrilatArea(cued_down_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,1,1,1] = quadrilatArea(cued_down_3Dcoords[n_colours*3:n_colours*4,:])
        
        if constants.PARAMS['experiment_number'] == 3:
            #post-probe
            CDI[model,1,2,0] = quadrilatArea(cued_down_3Dcoords[n_colours*4:n_colours*5,:])
            CDI[model,1,2,1] = quadrilatArea(cued_down_3Dcoords[n_colours*5:,:])
    # average across the trial types
    CDI_av = CDI.mean(1)
    # for pre-cue, average the cued and uncued
    if constants.PARAMS['experiment_number'] == 3:
        CDI_av = np.concatenate((np.expand_dims(CDI_av[:,0,:].mean(-1),-1),CDI_av[:,1,:],CDI_av[:,2,:]),1)
    else:
        CDI_av = np.concatenate((np.expand_dims(CDI_av[:,0,:].mean(-1),-1),CDI_av[:,1,:]),1)
    # CDI_av_df = pd.DataFrame(CDI_av,columns=['pre-cue','post_cued','post_uncued'])
    
    # # save structure
    # CDI_av_df.to_csv(load_path+'/CDI.csv')
    mean_PVE = PVEs.mean()*100
    sem_PVE = np.std(PVEs.mean(-1))*100 / np.sqrt(constants.PARAMS['n_models'])
    print('%% variance explained by 3D subspaces mean = %.2f, sem = %.3f' %(mean_PVE,sem_PVE))
    return CDI_av


#%%



