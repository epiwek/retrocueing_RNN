#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:07:28 2022

@author: emilia
"""
# from sklearn.decomposition import PCA
# from scipy.stats import zscore, stats, linregress, mode
# import statsmodels.api as sm
# import constants
# import pickle
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt


# load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
# model_path = constants.PARAMS['FULL_PATH'] + 'saved_models/'

# test_valid = pickle.load((open(load_path+'0.pckl','rb')))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for m in range(10):
#     # load output layer activations
#     f = open(load_path+'model_outputs_model'+str(m)+'.pckl','rb')
#     model_outputs = pickle.load(f)
#     f.close()
#     outputs = model_outputs['data']
    
    
    
    
    
#     f = open(load_path+'eval_data_model'+str(m)+'.pckl','rb')
#     eval_data = pickle.load(f)
#     f.close()
#     delay2 = eval_data['data'][:,-1,:]
    
#     data = delay2[:4,:].clone()
#     data -= data.mean()
#     # Initialise PCA object
#     pca = PCA(n_components=2) 
#     # get coordinates in the reduced-dim space
#     pca.fit(data)
    
#     loadings = pca.components_
    
#     pca2 = PCA(n_components=2) 
#     loadings2 = pca2.components_
    
#     constants.PARAMS['model_number'] = m
#     model = retnet.load_model(model_path,constants.PARAMS,device)
    
#     out_weights = model.out.weight.detach()
    
#     X = np.stack((test_valid['c1'],
#                   test_valid['c2'],
#                   test_valid['loc'][0,:,0],
#                   test_valid['c1']*test_valid['c2'],
#                   test_valid['c1']*test_valid['c2']*test_valid['loc'][0,:,0]),1)
    
    
#     X = zscore(X)
#     X = sm.add_constant(X)
    
    
    
#     coefs = np.empty((200,6))
#     for n in range(200):
#         data = delay2[:,n]
#         reg = LinearRegression().fit(X,data)
#         coefs[n,:] = reg.coef_
    
    
#     # plot betas
#     # c1 vs c2
    
#     plt.figure()
#     plt.plot(coefs[:,1],coefs[:,2],'o')
#     plt.title('c1 vs c2')
    
#     plt.figure()
#     plt.plot(coefs[:,1],coefs[:,-1],'o')
#     plt.title('c1 vs 3wint')
    
#     plt.figure()
#     plt.plot(coefs[:,2],coefs[:,-1],'o')
#     plt.title('c2 vs 3wint')


# #%%
# import seaborn as sns
# pca, coords = get_3D_coords(delay2)
# cols = sns.color_palette('husl',16)

# i = 0 # pick one of the 16 colours


# plt.figure()
# ax = plt.subplot(111,projection='3d')

# n_reps = coords.shape[0]//32
# col_spacing = n_reps//16
# colours=  np.unique(eval_data['labels']['c1'])




# half_split = eval_data['labels']['c2'].shape[0]//2

# # plot that colour when cued at loc1

# c2_switch_ix = np.where(np.diff(eval_data['labels']['c2'][:half_split+1])!=0)[0]+1
# c1_switch_ix = np.where(np.diff(eval_data['labels']['c1'][:half_split+1])!=0)[0]+1

# c2_switch_ix = np.concatenate(([0],c2_switch_ix))
# c1_switch_ix = np.concatenate(([0],c1_switch_ix))


# ix = c2_switch_ix[np.logical_and(c2_switch_ix>=c1_switch_ix[i],c2_switch_ix<c1_switch_ix[i+1])]
# ax.plot(coords[ix,0],
#         coords[ix,1],
#         coords[ix,2],
#         marker='o', c = cols[i])


# # plot that colour when cued at loc2

# c2_switch_ix = np.where(np.diff(eval_data['labels']['c2'][half_split+1:])!=0)[0]+1
# c1_switch_ix = np.where(np.diff(eval_data['labels']['c1'][half_split+1:])!=0)[0]+1

# c2_switch_ix = np.concatenate(([0],c2_switch_ix))
# c1_switch_ix = np.concatenate(([0],c1_switch_ix))
# c2_switch_ix += half_split+1
# c1_switch_ix += half_split+1


# ix = c1_switch_ix[np.logical_and(c1_switch_ix>=c2_switch_ix[i],c1_switch_ix<c2_switch_ix[i+1])]
    
# ax.plot(coords[ix,0],
#         coords[ix,1],
#         coords[ix,2],
#         marker='^', c = cols[i])
    
    
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')

# #%%

# # plot that colour when cued at loc1 and uncued at loc 1


# plt.figure()
# ax = plt.subplot(111,projection='3d')
# c2_switch_ix = np.where(np.diff(eval_data['labels']['c2'][:half_split+1])!=0)[0]+1
# c1_switch_ix = np.where(np.diff(eval_data['labels']['c1'][:half_split+1])!=0)[0]+1

# c2_switch_ix = np.concatenate(([0],c2_switch_ix))
# c1_switch_ix = np.concatenate(([0],c1_switch_ix))
# ix = c2_switch_ix[np.logical_and(c2_switch_ix>=c1_switch_ix[i],c2_switch_ix<c1_switch_ix[i+1])]

# ax.plot(coords[ix,0],
#         coords[ix,1],
#         coords[ix,2],
#         marker='o', c = cols[i],label='cued')


# # uncued
# c1_switch_ix = np.where(np.diff(eval_data['labels']['c1'][half_split:])!=0)[0]+1
# c1_switch_ix = np.concatenate(([0],c1_switch_ix))
# c1_switch_ix += half_split
# c2_switch_ix = np.where(np.diff(eval_data['labels']['c2'][half_split:])!=0)[0]+1
# c2_switch_ix = np.concatenate(([0],c2_switch_ix))
# c2_switch_ix += half_split

# ix = np.intersect1d(c1_switch_ix,c2_switch_ix)
# ax.plot(coords[ix,0],
#         coords[ix,1],
#         coords[ix,2],
#         marker='s', c = cols[i],label='uncued')


#%% do some decoding


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from scipy import sparse
import pandas as pd
from itertools import groupby, combinations
import numpy as np
import custom_plot as cplt 
import helpers
import constants
import pickle
import matplotlib.pyplot as plt



def get_class_pairs(y):
    classes = np.unique(y)
    combos = list(combinations(classes,2))
    return combos

#%
def bin_labels(labels_list,n_bins):
    labels = np.unique(labels_list)
    n_labels = len(labels)
    
    if (n_labels % n_bins != 0):
        raise ValueError('N labels must be divisible by N bins')
    
    bin_over = n_labels//n_bins
    binned_labels = np.array([np.arange(n_bins)]*bin_over).reshape(-1,order='F')
        
    labels_list_integer = np.array([np.where(labels==val)[0] \
                            for ix,val in enumerate(np.array(labels_list))])
    labels_list_binned = np.squeeze(np.array([binned_labels[val] \
                            for ix,val in enumerate(np.array(labels_list_integer))]))
        
    return labels_list_binned


def lda(X,y, cv=2):
    # split into train and test
    
    class_combos= get_class_pairs(y)
    scores_test = np.zeros((len(class_combos)))
    for i in range(len(class_combos)):
        y1,y2 = class_combos[i] # class labels
        
        # find trials of the abovespecified classes
        trial_ix = np.where(np.logical_or(y==y1,y==y2))[0]
        
        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LDA())
        
        # fit a classifier in cross-val
        results = cross_validate(clf,X[trial_ix,:],y[trial_ix],cv=cv,return_estimator=False)
        
        # average test scores across cv folds
        scores_test[i] = results['test_score'].mean()
    
    return scores_test     


def lda_cg(X1, X2, y1, y2, cv=2):
    class_combos= get_class_pairs(y1)
    clfs = {}
    scores_test = np.empty((len(class_combos),2))
    scores_cg = np.empty((len(class_combos),2))
    for i in range(len(class_combos)):
        c1,c2 = class_combos[i] # class labels
        # make pipeline
        clf = make_pipeline(StandardScaler(), LDA())
        
        y1_ix = np.where(np.logical_or(y1==c1,y1==c2))[0]
        y2_ix = np.where(np.logical_or(y2==c1,y2==c2))[0]
    
    
        # fit a classifier in cross-val
        clfs[str(i)] = cross_validate(clf,X1[y1_ix,:],y1[y1_ix],cv=cv,return_estimator=True)
        
        # find the best-performing one
        # best_ix = np.where(clfs[str(i)]['test_score']==np.max(clfs[str(i)]['test_score']))[0][0]
        
        scores_test[i,:] = clfs[str(i)]['test_score']
        # test on with-held data (cross-gen)
        for c in range(cv):
            scores_cg[i,c] = clfs[str(i)]['estimator'][c].score(X2[y2_ix,:],y2[y2_ix])
    
        
    return np.mean(scores_test), np.mean(scores_cg), clfs

#%% train on cued, test on uncued - both locations

load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'

model_scores_test = np.empty((constants.PARAMS['n_models'],))
model_scores_cg = np.empty((constants.PARAMS['n_models'],))


for model in range(constants.PARAMS['n_models']):
    # load training data
    model_number = str(model)
    print('Model '+model_number)
    f = open(load_path+'eval_data_model' + model_number + '.pckl', 'rb')
    eval_data = pickle.load(f)    
    f.close()
    
    n_trials = eval_data['data'].shape[0]
    
    labels_cued = np.concatenate((eval_data["labels"]["c1"][:n_trials//2],
                              eval_data["labels"]["c2"][n_trials//2:]))
    labels_cued = helpers.bin_labels(labels_cued,constants.PARAMS['B'])
    
    trial_ix = np.arange(n_trials)
    # probe_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                              eval_data["labels"]["c1"][n_trials//2:]))
    labels_uncued = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
    
    # probeloc1 = eval_data['data'][:n_trials//2,-1,:]
    # probeloc2 = eval_data['data'][n_trials//2:,-1,:]
    probe_data = eval_data['data'][:,-1,:]
    
    # loc1_labels = labels_cued[:n_trials//2]
    # loc2_labels = labels_cued[n_trials//2:]
    # trial_ix = np.arange(n_trials//2)
    X_train,X_test,train_ix,test_ix= train_test_split(probe_data,trial_ix,test_size=.2)
    
    y_train = labels_cued[train_ix]
    y_test = labels_uncued[test_ix]
    
    
    # do LDA
    model_scores_test[model], model_scores_cg[model],clfs = lda_cg(X_train,X_test,y_train,y_test)   
    
    
#%% train on cued, test on uncued - same location

model_scores_test = np.empty((constants.PARAMS['n_models'],2))
model_scores_cg = np.empty((constants.PARAMS['n_models'],2))

for model in range(constants.PARAMS['n_models']):
    # load training data
    model_number = str(model)
    print('Model '+model_number)

    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    eval_data = pickle.load(f)    
    f.close()
    
    n_trials = eval_data['data'].shape[0]
    
    labels_cued = np.concatenate((eval_data["labels"]["c1"][:n_trials//2],
                              eval_data["labels"]["c2"][n_trials//2:]))
    labels_cued = helpers.bin_labels(labels_cued,constants.PARAMS['B'])
    
    # trial_ix = np.arange(n_trials)
    # probe_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                              eval_data["labels"]["c1"][n_trials//2:]))
    labels_uncued = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
    
    probeloc1 = eval_data['data'][:n_trials//2,-1,:]
    probeloc2 = eval_data['data'][n_trials//2:,-1,:]
    # probe_data = eval_data['data'][:,-1,:]
    
    # loc1_labels = labels_cued[:n_trials//2]
    # loc2_labels = labels_cued[n_trials//2:]
    trial_ix = np.arange(n_trials//2)
    X_train,_,train_ix,test_ix= train_test_split(probeloc1,trial_ix,test_size=.2)
    
    y_train = labels_cued[train_ix]
    X_test = probeloc2[test_ix,:]
    y_test = labels_uncued[test_ix+n_trials//2]
    
    # do LDA
    model_scores_test[model,0],model_scores_cg[model,0],clfs = lda_cg(X_train,X_test,y_train,y_test) # loc1
    
    trial_ix = np.arange(n_trials//2,n_trials)
    X_train,_,train_ix,test_ix= train_test_split(probeloc2,trial_ix,test_size=.2)
    
    y_train = labels_cued[train_ix]
    X_test = probeloc1[test_ix-n_trials//2,:]
    y_test = labels_uncued[test_ix]
    
    
    model_scores_test[model,1],model_scores_cg[model,1],clfs = lda_cg(X_train,X_test,y_train,y_test) # loc2