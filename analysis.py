#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:42:36 2021

@author: emilia
"""
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS


def make_rdm(data,metric='correlation'):
    rdm = squareform(pdist(data,metric))
    return rdm



def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)
