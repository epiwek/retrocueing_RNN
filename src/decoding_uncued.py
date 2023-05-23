#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:58:34 2021

@author: emilia
"""

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



#%%

def get_freqs(x):
    """
    Computes the ocurence frequency of each class of objects
    
    :param x list of labels:

    :return dict with every object and conresponding dictionary:

    """
    return {value: len(list(freq)) for value, freq in groupby(sorted(list(x)))}


def equalise_classes(X, y, shuffle=False, min_freq=None):
    """ Re-samples the data array and label vector to match the frequencies of object classes

        X: trials x neurons x time
        y: 1d label list
        shuffle: if True scramble to trial-label affiliation
        min_freq: if other than None sets the minimal frequency for all classes

        :return re-sampled data array
                re-sampled label list
                determined minimal frequency

    """

    results = get_freqs(y)

    if min_freq == None:
        min_val = min(results.values())
    else:
        min_val = int(min_freq)

    df_idcs = pd.DataFrame(y,columns=['class'])
    df_idcs['Indices'] = list(range(0,len(y)))

    idc_lis = []
    for _ in range(0,len(results.keys())):
        idc_lis.append(df_idcs[df_idcs['class'] == float(_+1)].sample(min_val))

    df_idcs_res = pd.concat(idc_lis, axis=0)
    if shuffle:
        idc_res = df_idcs_res['Indices'].values
    else:
        idc_res = df_idcs_res['Indices'].values
        idc_res.sort()

    return X[idc_res, :, :], y[idc_res], min_val

# def lda_sliding(X, y, cv =3, time_dim=False):
    # X - trials x neurons x time (optional)
    # y - labels list

    # clf = make_pipeline(StandardScaler(), LDA())

    # if time_dim:
    #     classifier = SlidingEstimator(clf)
    # else:
    #     classifier = clf

    # Run cross-validated decoding:
    # scores = cross_val_multiscore(classifier, X, y, cv=StratifiedKFold(n_splits=cv))
    # return scores.mean(0)

# def lda(X,y, cv=2):
#     # split into train and test
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)
    
#     # make pipeline
#     clf = make_pipeline(StandardScaler(), OneVsOneClassifier(LDA(priors=[.5,.5])))

#     # do cross-validation on the dataset coming from one delay - to assess it's performance
#     # on the plane it was trained on
#     cv_clf = cross_validate(clf,X_train,y_train,cv=cv,return_estimator=True)
    
    
#     best_ix = np.where(cv_clf['test_score']==np.max(cv_clf['test_score']))[0][0]
#     scores_Xtrain = cv_clf['test_score'][best_ix]
    
#     # additionally, test the classifier on the dataset from the other plane - to test for alignment
#     # indiv_scores_X2 = []
#     # for e in range(cv):
#     #     indiv_scores_X2.append(cv_clf['estimator'][e].score(X2,y2))
#     # scores_X2 = np.mean(indiv_scores_X2)
    
#     scores_Xtest = cv_clf['estimator'][best_ix].score(X_test,y_test)
    
#     return scores_Xtrain, scores_Xtest, cv_clf

# def lda(X,y, cv=2):
#     # split into train and test
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)
    
#     class_combos= get_class_pairs(y)
#     clfs = {}
#     scores_Xtest = np.zeros(len(class_combos,))
#     for i in range(len(class_combos)):
#         y1,y2 = class_combos[i] # class labels
#         # make pipeline
#         clf = make_pipeline(StandardScaler(), LDA())
        
#         train_ix = np.where(np.logical_or(y_train==y1,y_train==y2))[0]
#         test_ix = np.where(np.logical_or(y_test==y1,y_test==y2))[0]
    
    
#         # fit a classifier in cross-val
#         clfs[str(i)] = cross_validate(clf,X_train[train_ix,:],y_train[train_ix],cv=cv,return_estimator=True)
        
#         # find the best-performing one
#         best_ix = np.where(clfs[str(i)]['test_score']==np.max(clfs[str(i)]['test_score']))[0][0]
        
        
#         # test on with-held data
#         scores_Xtest[i] = clfs[str(i)]['estimator'][best_ix].score(X_test[test_ix,:],y_test[test_ix])
    
        
#     return np.mean(scores_Xtest), clfs


# def lda(X,y, cv=2):
#     # split into train and test
    
#     class_combos= get_class_pairs(y)
#     clfs = {}
#     scores_test = np.zeros((len(class_combos),2))
#     for i in range(len(class_combos)):
#         y1,y2 = class_combos[i] # class labels
        
#         # find trials of abovespecified classes
#         trial_ix = np.where(np.logical_or(y==y1,y==y2))[0]
        
#         # make classifier pipeline
#         clf = make_pipeline(StandardScaler(), LDA())
        
#         # fit a classifier in cross-val
#         clfs[str(i)] = cross_validate(clf,X[trial_ix,:,:],y[trial_ix],cv=cv,return_estimator=True)
        
#         # find the best-performing one
#         best_ix = np.where(clfs[str(i)]['test_score']==np.max(clfs[str(i)]['test_score']))[0][0]
        
        
#         # test on with-held data
#         scores_test[i] = clfs[str(i)]['test_score']
#         #clfs[str(i)]['estimator'][best_ix].score(X_test[test_ix,:,:],y_test[test_ix])
    
#     return np.mean(scores_test)
# # #%% normal decoding

# # X = ...
# # y = ...

# # X_re, y_re, _ = equalise_classes(X,y)
# # score = lda_sliding(X_re, y_re, time_dim=True)


# def lda_cg_only(cv_clf,X_test,y_test):
#     # take a list of pre-trained classifiers from crossval
#     # find the best-performing one
#     # apply it to test data (cross-generalisation)
    
#     class_combos= get_class_pairs(y_test)
#     scores = np.zeros((len(class_combos),))
#     for i in range(len(class_combos)):
#         y1,y2 = class_combos[i] # class labels
#         class_ix = np.where(np.logical_or(y_test==y1,y_test==y2))[0]
        
#         best_ix = np.where(cv_clf[str(i)]['test_score']==np.max(cv_clf[str(i)]['test_score']))[0][0]
#         scores[i] = cv_clf[str(i)]['estimator'][best_ix].score(X_test[class_ix,:],y_test[class_ix])
    
#     return np.mean(scores)

# # def lda_cg_only(cv_clf,X_test,y_test):
# #     # take a list of pre-trained classifiers from crossval
# #     # find the best-performing one
# #     # apply it to test data (cross-generalisation)
    
# #     best_ix = np.where(cv_clf['test_score']==np.max(cv_clf['test_score']))[0][0]
# #     scores = cv_clf['estimator'][best_ix].score(X_test,y_test)
# #     return scores


# # def lda_cg(X1, X2, y1, y2, cv=3):
# #     # cross gen
# #     # X1 - train
# #     # X2 - test
# #     clf = make_pipeline(StandardScaler(), OneVsOneClassifier(LDA()))

# #     # do cross-validation on the dataset coming from one delay - to assess it's performance
# #     # on the plane it was trained on
# #     cv_clf = cross_validate(clf,X1,y1,cv=cv,return_estimator=True)
    
    
# #     best_ix = np.where(cv_clf['test_score']==np.max(cv_clf['test_score']))[0][0]
# #     scores_X1 = cv_clf['test_score'][best_ix]
    
# #     # additionally, test the classifier on the dataset from the other plane - to test for alignment
# #     # indiv_scores_X2 = []
# #     # for e in range(cv):
# #     #     indiv_scores_X2.append(cv_clf['estimator'][e].score(X2,y2))
# #     # scores_X2 = np.mean(indiv_scores_X2)
    
# #     scores_X2 = cv_clf['estimator'][best_ix].score(X2,y2)
    
# #     return scores_X1, scores_X2, cv_clf

# def lda_cg(X1, X2, y1, y2, cv=3):
#     class_combos= get_class_pairs(y1)
#     clfs = {}
#     scores_Xtest = np.zeros(len(class_combos,))
#     for i in range(len(class_combos)):
#         c1,c2 = class_combos[i] # class labels
#         # make pipeline
#         clf = make_pipeline(StandardScaler(), LDA())
        
#         y1_ix = np.where(np.logical_or(y1==c1,y1==c2))[0]
#         y2_ix = np.where(np.logical_or(y2==c1,y2==c2))[0]
    
    
#         # fit a classifier in cross-val
#         clfs[str(i)] = cross_validate(clf,X1[y1_ix,:],y1[y1_ix],cv=cv,return_estimator=True)
        
#         # find the best-performing one
#         best_ix = np.where(clfs[str(i)]['test_score']==np.max(clfs[str(i)]['test_score']))[0][0]
        
        
#         # test on with-held data
#         scores_Xtest[i] = clfs[str(i)]['estimator'][best_ix].score(X2[y2_ix,:],y2[y2_ix])
    
        
#     return np.mean(scores_Xtest), clfs

def get_class_pairs(y):
    classes = np.unique(y)
    combos = list(combinations(classes,2))
    return combos

#%%
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
#%% load data

# import pickle
# # base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian_hardcoded/nrec300/lr0.005/'
# base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
#     'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/'



# load_path = base_path + 'pca_data'
# f = open(load_path+'/converged.pckl','rb')
# converged = pickle.load(f)
# f.close()

# n_models = len(converged)

# #%%
# scores = np.zeros((n_models,3))
# scores_cg = np.zeros((n_models,3))


# for model_number in range(10):
#     i = model_number
#     model_number = str(model_number)

#     # load eval data
#     f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
#     obj = pickle.load(f)
#     data = obj["data"]
#     f.close()
    
    
#     pre_cue = data[:,1,:]
#     post_cue = data[:,3,:]
#     # n_colours = int(np.sqrt(data.shape[0]//2))
#     n_colours = 4
    
#     n_trials = pre_cue.shape[0]//(2*n_colours)
#     c1_cued_ix = np.arange(post_cue.shape[0]//2)
#     c2_cued_ix = np.arange(post_cue.shape[0]//2,post_cue.shape[0])
    

    
    
#     # X1 = pre_cue[:post_cue.shape[0]//2,:]                          # loc 1
#     # X2 = pre_cue[post_cue.shape[0]//2:,:]                          # loc 2
#     X1 = post_cue[c1_cued_ix,:]                          # plane 1
#     X2 = post_cue[c2_cued_ix,:]                          # plane 2
#     # y1 = np.array([np.arange(n_colours)]*n_trials).reshape(-1,order='F')
#     # y2 = y1
    
#     y1 = obj["labels"]['c2'][c1_cued_ix]
#     y1_binned = bin_labels(y1,n_colours)
    
#     y2 =  obj["labels"]['c1'][c2_cued_ix]
#     y2_binned = bin_labels(y2,n_colours)
#     # scores_X1train, scores[i,0], cv_clf1 = lda(X1,y1_binned)
#     # scores_X2train, scores[i,1], cv_clf2 = lda(X2,y2_binned)
    
#     scores[i,0], cv_clf1 = lda(X1,y1_binned)
#     scores[i,1], cv_clf2 = lda(X2,y2_binned)
    
#     scores_cg[i,0] = lda_cg_only(cv_clf1,X2,y2_binned)
#     scores_cg[i,1] = lda_cg_only(cv_clf2,X1,y1_binned)
    
    
# scores[:,-1] = np.mean(scores[:,:2],1)
# scores_cg[:,-1] = np.mean(scores_cg[:,:2],1)


# #%% stats

# from scipy.stats import shapiro, ttest_1samp, wilcoxon



# chance_lvl = 0.5

# print('Decoding accuracy post-cue, same plane:')
# sw, p = shapiro(scores[:,-1])

# if p<0.05:
#     print('    wilcoxon test')
#     stat, p_val = wilcoxon(scores[:,-1],np.ones(10)*chance_lvl)
# else:
#     print('    one-sample t-test')
#     stat, p_val = ttest_1samp(scores[:,-1],chance_lvl)
# print('        stat = %.3f, p = %.3f' %(stat,p_val))

# #%%

# print('Decoding accuracy post-cue, other plane:')

# sw, p = shapiro(scores_cg[:,-1])


# if p<0.05:
#     print('    wilcoxon test')
#     stat, p_val = wilcoxon(scores_cg[:,-1],np.ones(10)*chance_lvl)
# else:
#     print('    one-sample t-test')
#     stat, p_val = ttest_1samp(scores_cg[:,-1],chance_lvl)
# print('        stat = %.3f, p = %.3f' %(stat,p_val))
    
# #%% plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# import custom_plot as cplt 

# plt.rcParams.update({'font.size': 30})
# plt.figure(figsize=(10,10))
# ax = plt.subplot(111)

# pal = sns.color_palette("dark")
# inds = [3,0]
# cols = [pal[ix] for ix in inds]
# ms=16


# mean_scores = [np.mean(scores[:,-1]*100),np.mean(scores_cg[:,-1]*100)]
# # labels = np.concatenate((np.zeros((n_models,)),np.ones((n_models,))))


# scores_tbl = np.stack((scores[:,-1],scores_cg[:,-1])).T*100

# plt.bar([0,1],mean_scores,width=.25,color =cols,alpha=.2)
# # ax = sns.stripplot(x=labels,y=scores,palette=cols,size=8)

# # for model in range(n_models):
# #     plt.plot(['pre','post'],scores[model,:],'k-',alpha=.2)
# #     plt.plot(['pre'],scores[model,0],'o',c=cols[0],markersize=ms)
# #     plt.plot(['post'],scores[model,1],'o',c=cols[1],markersize=ms)

# xs = np.zeros((n_models,2))
# xs[:,1] = 1
# cplt.plot_paired_data(xs, scores_tbl, ax, cols,jitter=.0001,markersize=ms)   

# plt.ylabel('Decoding accuracy [%]')

# plt.ylim(0, 100)

# plt.xlim(-.5,1.5)
# xlims = plt.xlim()
# plt.plot(xlims,np.ones((2,))*chance_lvl*100,'k--',label='chance level')
# # plt.xlim(xlims)
# plt.xticks([0,1],['same','other'])

# plt.legend()


# plt.xlabel('Location')

# plt.tight_layout()

# # add stats
# # ax.plot(0,98,'w*',markersize=20)

# #%% decode SI vs delay1
# import torch
# n_models=10
# scores_cg = np.zeros((n_models,3,2)) # delay1inSI, SIindelay1, score_mean /loc1-2

# # scores_post_post = np.zeros((n_models,3)) # score1, score2, score_mean
# # scores_post_pre = np.zeros((n_models,3)) # score1, score2, score_mean

# # import torch

# # for i,model_number in enumerate(converged):
# for model_number in range(10):
#     i = model_number
#     model_number = str(model_number)
    

#     # load RDM data
#     f = open(load_path+'/rdm_data_model' + model_number + '.pckl', 'rb')
#     obj = pickle.load(f)
#     data = obj
#     f.close()
    
#     # load eval data
#     # f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
#     # obj = pickle.load(f)
#     # data = obj["data"]
#     # f.close()
    
    
#     SI = data[:,0,:]
#     pre_cue = data[:,1,:]
    
#     # n_colours = int(np.sqrt(data.shape[0]//2))
#     n_colours = 4
    
#     n_trials = SI.shape[0]//(2*n_colours)
    
#     X1 = SI[:SI.shape[0]//2,:] #S1 loc1
#     X2 = SI[SI.shape[0]//2:,:] #S1 loc2
#     y1 = np.array([np.arange(n_colours)]*n_trials).reshape(-1,order='F')
#     y2 = y1
    
#     X3 = pre_cue[:pre_cue.shape[0]//2,:] # pre-cue loc1
#     X4 = pre_cue[pre_cue.shape[0]//2:,:] # pre-cue loc2
    
    
    
#     # X1 = pre_cue[:16,:]
#     # X2 = pre_cue[16:,:]
#     # y1 = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
#     # y2 = y1
#     # scores_pre[i,0] = lda_cg(X1, X2, y1, y2, time_dim=False)
#     # scores_pre[i,1] = lda_cg(X2, X1, y2, y1, time_dim=False)
    
#     # tmp, scores_cg[i,0,0], cv_clf = lda_cg(X1, X3[torch.randperm(16),:], y1, y2)
#     # tmp, scores_cg[i,1,0], cv_clf = lda_cg(X3, X1[torch.randperm(16),:], y1, y2)
#     # tmp, scores_cg[i,0,1], cv_clf = lda_cg(X2, X4[torch.randperm(16),:], y1, y2)
#     # tmp, scores_cg[i,1,1],cv_clf = lda_cg(X4, X2[torch.randperm(16),:], y2, y1)
    
    
#     scores_cg[i,0,0], cv_clf = lda_cg(X1, X3, y1, y2,cv=2)
#     scores_cg[i,1,0], cv_clf = lda_cg(X3, X1, y1, y2,cv=2)
#     scores_cg[i,0,1], cv_clf = lda_cg(X2, X4, y1, y2,cv=2)
#     scores_cg[i,1,1],cv_clf = lda_cg(X4, X2, y2, y1,cv=2)
    
    
    
    
# #%% print table

# scores_cg[:,2,:] = np.mean(scores_cg[:,:2,:],axis=1)

# scores_d1inSI = np.mean(scores_cg[:,0,:],-1)
# scores_SIind1 = np.mean(scores_cg[:,1,:],-1)
# #%% plots

# import matplotlib.pyplot as plt
# import seaborn as sns
# import custom_plot as cplt

# plt.rcParams.update({'font.size': 30})
# plt.figure(figsize=(10,10))
# ax = plt.subplot(111)

# pal = sns.color_palette("dark")
# inds = [3,0]
# cols = [pal[ix] for ix in inds]
# ms=16


# mean_scores = [np.mean(scores_d1inSI*100),np.mean(scores_SIind1*100)]
# # labels = np.concatenate((np.zeros((n_models,)),np.ones((n_models,))))


# scores_tbl = np.stack((scores_d1inSI,scores_SIind1)).T*100

# plt.bar([0,1],mean_scores,width=.25,color =cols,alpha=.2)
# # ax = sns.stripplot(x=labels,y=scores,palette=cols,size=8)

# # for model in range(n_models):
# #     plt.plot(['pre','post'],scores[model,:],'k-',alpha=.2)
# #     plt.plot(['pre'],scores[model,0],'o',c=cols[0],markersize=ms)
# #     plt.plot(['post'],scores[model,1],'o',c=cols[1],markersize=ms)

# xs = np.zeros((n_models,2))
# xs[:,1] = 1
# cplt.plot_paired_data(xs, scores_tbl, ax, cols,jitter=.0001,markersize=ms)   

# plt.ylabel('Decoding accuracy [%]')

# plt.ylim(0, 100)

# plt.xlim(-.5,1.5)
# xlims = plt.xlim()
# plt.plot(xlims,np.ones((2,))*chance_lvl*100,'k--',label='chance level')
# # plt.xlim(xlims)
# plt.xticks([0,1],['d1inSI','SIind1'])

# plt.legend()


# plt.xlabel('Phantom',c='w')

# plt.tight_layout()




#%% decode uncued in post-cue - is the information still there?
import constants
import pickle
import helpers
load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'


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


model_scores = np.empty((constants.PARAMS['n_models'],))
for model in range(constants.PARAMS['n_models']):
    # load data
    model_number = str(model)
    print('Model '+ model_number)
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    eval_data = pickle.load(f)    
    f.close()
    
    n_trials = eval_data['data'].shape[0]
    delay2_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                              eval_data["labels"]["c1"][n_trials//2:]))
    labels_uncued_binned = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
    loc1_labels = labels_uncued_binned[:n_trials//2]
    loc2_labels = labels_uncued_binned[n_trials//2:]
    
    delay2loc1 = eval_data['data'][:n_trials//2,delay2_ix,:]
    delay2loc2 = eval_data['data'][n_trials//2:,delay2_ix,:]
    
    # do LDA
    scores_loc1 = lda(delay2loc1,loc1_labels)
    scores_loc2 = lda(delay2loc2,loc2_labels)
    
    model_scores[model] = np.stack((scores_loc1,scores_loc2)).mean()
        
# import seaborn as sns
# plt.figure(figsize=(3.75,4.8))
# sns.boxplot(data=model_scores,orient='v',color=[.5,.5,.5])
# plt.ylim([.45,1.0])
# x_vals = np.linspace(plt.xlim()[0],plt.xlim()[1])
# y_vals = np.ones(len(x_vals))*.5
# plt.plot(x_vals,y_vals,'k--')
# plt.ylabel('Decoding accuracy')
# plt.xticks([])
# plt.tight_layout()
# plt.savefig(constants.PARAMS['FIG_PATH']+'uncued_decoding.png')


# test against chance

from scipy.stats import shapiro, wilcoxon, ttest_1samp

_, p = shapiro(model_scores)

if p < .05:
    stat,p = wilcoxon(model_scores,y=.5,alternative='greater')
    print('Wilcoxon test: W = %.3f, p = %.3f' %(stat,p))
else:
    stat,p = ttest_1samp(model_scores,.5,alternative='greater')
    print('1-sample t-test: t = %.3f, p = %.3f' %(stat,p))
    
    
print('Mean performance: %.4f' %model_scores.mean())
