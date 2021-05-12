# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:24:01 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import itertools as itr
import pandas as pd
from bct.algorithms import(centrality, clustering, distance, modularity, core, similarity)

from reservoir.network import nulls

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
 
#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA 
# ----------------------------------------------------------------------------------------------------------------------
CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability/scale500'
#NULL_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/significance/scale500'

ctx = np.load('E:/P3_RC/neuromorphic_networks/data/cortical/cortical_human_500.npy')

rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

labels, counts = np.unique(rsn_mapp_int, return_counts=True)

coords = np.load('E:/P3_RC/neuromorphic_networks/data/coords/coords_human_500.npy')    
conn = np.load(os.path.join(CONN_DIR, 'consensus_0.npy')) 

#%% --------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def heatmap(data, annot, title=None, **kwargs):
    
    sns.set(style="ticks", font_scale=2.0) 
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(111)
    
    sns.heatmap(data, 
#                vmin=0.0, vmax=3.0, 
    #            center=0.75, 
                cmap=sns.cubehelix_palette(as_cmap=True), #ListedColormap(sns.color_palette('cubehelix')),
                square=True, 
                xticklabels=rsn_labels, yticklabels=rsn_labels,
    #                mask=mask.astype(bool),  # ~distance.astype(bool), 
                annot=np.round(annot,2),
                linewidths=0.2,
                annot_kws={'fontsize':25, 'fontweight':'bold', 'color':'white'}, #, 'color':'white'},##eed8c9'},
                fmt='g',
                linecolor='white',
                **kwargs
                )
    
    plt.tick_params(axis='both', which='major', labelsize=20, labelbottom = False, bottom=False, 
                    top = True, labeltop=True)
    
    plt.setp(ax.get_xticklabels(), rotation=45, 
             #ha="right",
             #rotation_mode="anchor"
             )
    
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor"
             )
    
    #sns.despine(offset=10, left=True, trim=True)
    if title is not None: fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs', f'{title}.png'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()


#%% --------------------------------------------------------------------------------------------------------------------
# DEGREE AND STRENGTH
# ----------------------------------------------------------------------------------------------------------------------
strength = np.zeros((len(rsn_labels), len(rsn_labels)))
degree = np.zeros((len(rsn_labels), len(rsn_labels)))

for a,b in itr.product(rsn_labels, rsn_labels):
    
    idx_a = np.where(rsn_labels == a)[0][0]
    idx_b = np.where(rsn_labels == b)[0][0]
    
    tmp_conn = conn.copy()[np.ix_(np.where(rsn_mapp == a)[0], np.where(rsn_mapp == b)[0])]
    
    if a == b: degree[idx_a,idx_b] = (np.sum(tmp_conn.astype(bool).astype(int)))/2
    else: degree[idx_a,idx_b] = (np.sum(tmp_conn.astype(bool).astype(int)))

    if a == b: strength[idx_a,idx_b] = (np.sum(tmp_conn))/2
    else: strength[idx_a,idx_b] = (np.sum(tmp_conn))
    
#%%
#degree[np.tril_indices_from(degree, -1)] = 0 
n_degree = degree.copy()/np.sum(degree, axis=1)[:,np.newaxis]
#heatmap(data=n_degree, annot=n_degree, title='norm_degree')

#%%
#strength[np.tril_indices_from(strength, -1)] = 0 
n_strength = strength.copy()/np.sum(strength, axis=1)[:,np.newaxis]
#heatmap(data=n_strength, annot=n_strength, title='norm_strength')

#%% --------------------------------------------------------------------------------------------------------------------
# RECIPROCITY
# ----------------------------------------------------------------------------------------------------------------------
var = n_degree.copy()

recip = np.zeros((len(rsn_labels), len(rsn_labels)))
for a,b in itr.product(rsn_labels, rsn_labels):
 
    idx_a = np.where(rsn_labels == a)[0][0]
    idx_b = np.where(rsn_labels == b)[0][0]

    if (a!=b) and (recip[idx_a,idx_b] == 0): 
        
        recip[idx_a, idx_b] = ((var[idx_a,idx_b]+var[idx_b,idx_a])/2)/(np.abs(var[idx_a, idx_b] - var[idx_b, idx_a]))
        recip[idx_b, idx_a] = recip[idx_a, idx_b]
    
    elif (a==b): 
        recip[idx_a,idx_b] = 0

heatmap(data=recip, annot=recip, title='reciprocity', vmin=0, vmax=20)
        
#%% --------------------------------------------------------------------------------------------------------------------
# SHORTEST PATH 
# ----------------------------------------------------------------------------------------------------------------------
#def char_path(sp):
#    
#    char_path = np.zeros((len(rsn_labels), len(rsn_labels)))
#    
#    for a,b in itr.product(rsn_labels, rsn_labels):
#        
#        idx_a = np.where(rsn_labels == a)[0][0]
#        idx_b = np.where(rsn_labels == b)[0][0]
#        
#        tmp_sp = sp.copy()[np.ix_(np.where(rsn_mapp == a)[0], np.where(rsn_mapp == b)[0])]
#        
#        if a == b: 
#            char_path[idx_a,idx_b] = np.mean(tmp_sp)/2
#        
#        else: 
#            char_path[idx_a,idx_b] = np.mean(tmp_sp)
#    
#    return char_path

#sp_wei, paths = distance.distance_wei(1/conn)
#np.save('C:/Users/User/Desktop/shortest_path_wei.npy', sp_wei)
#np.save('C:/Users/User/Desktop/paths.npy', paths)

#sp_bin = distance.distance_bin(conn.astype(bool).astype(int))
#np.save('C:/Users/User/Desktop/shortest_path_bin.npy', sp_bin)

#sp_wei = np.load('C:/Users/User/Desktop/paths.npy')
#sp_bin = np.load('C:/Users/User/Desktop/shortest_path_bin.npy')

#%%
#char_path_wei = char_path(sp_wei)
#heatmap(data=char_path_wei, 
#        annot=char_path_wei, 
#        title='char_path_wei',
#        vmin=0.0, vmax=8.0, 
#        )
#
#char_path_bin = char_path(sp_bin)
#heatmap(data=char_path_bin, 
#        annot=char_path_bin, 
#        title='char_path_bin',
#        vmin=0.0, vmax=3.0, 
#        )

