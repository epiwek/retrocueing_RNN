#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 01:26:13 2021

@author: emilia
"""

import pickle
base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/'

f = open(base_path+'RDM_analysis/indiv_model_rdms.pckl','rb')
indiv_model_rdms = pickle.load(f)

[rdm_precue_binned,rdm_postcue_binned] = indiv_model_rdms
f.close()


#%% get a projection of 3d coordinates onto the 2d planes

for model in range(10):
    f = open(base_path+'pca_data/planes/planes_RNN'+str(model)+'.pckl','rb')
    planes = pickle.load(f)
    
    [t,tt,plane1, plane2] = planes

