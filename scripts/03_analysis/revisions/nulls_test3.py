# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:56:46 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from math import log
from sklearn.metrics.cluster import (normalized_mutual_info_score, adjusted_mutual_info_score)

CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability_mod/scale500'
NULL_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/significance_mod/scale500'

ctx = np.load('E:/P3_RC/neuromorphic_networks/data/cortical/cortical_human_500.npy')
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

labels, counts = np.unique(rsn_mapp_int, return_counts=True)

#%%
def get_communities(class_mapping):
    
    return [np.where(class_mapping == label)[0] for label in np.unique(class_mapping)]

#%%
rsn_comm = get_communities(rsn_mapp_int)

nomr_MI_null = []
adj_MI_null = []
# rewired brain networks 
for i in range(1000):
    
    print(f'-------{i}-------')
    
    conn = np.load(f'{NULL_DIR}/rand_mio_{i}.npy')    
    
    opt_mapping = np.zeros(len(conn))
    opt_mapping[ctx == 1] = np.load(f'{NULL_DIR}/class_mapping_{i}.npy')
    opt_mapping[ctx == 0] = np.max(opt_mapping)+1
    opt_mapping = opt_mapping.astype(int)    

#    opt_comm = get_communities(opt_mapping)
        
    nomr_MI_null.append(normalized_mutual_info_score(rsn_mapp_int, opt_mapping))
    adj_MI_null.append(adjusted_mutual_info_score(rsn_mapp_int, opt_mapping))
    
#%%
# rewired brain networks 
rsn_comm = get_communities(rsn_mapp_int)

nomr_MI_emp = []
adj_MI_emp = []
for i in range(1000):
    
    print(f'-------{i}-------')
    
    conn = np.load(f'{NULL_DIR}/rand_mio_{i}.npy')    
    
    opt_mapping = np.zeros(len(conn))
    opt_mapping[ctx == 1] = np.load(f'{CONN_DIR}/class_mapping_{i}.npy')
    opt_mapping[ctx == 0] = np.max(opt_mapping)+1
    opt_mapping = opt_mapping.astype(int)    

#    opt_comm = get_communities(opt_mapping)
    
    nomr_MI_emp.append(normalized_mutual_info_score(rsn_mapp_int, opt_mapping))
    adj_MI_emp.append(adjusted_mutual_info_score(rsn_mapp_int, opt_mapping))


#%% REWIRED NETWORKS
rsn_comm = get_communities(rsn_mapp_int)

rsn_mod = []
opt_mod = []
for i in range(1000):
    
    print(i)
    
    conn = np.load(f'{NULL_DIR}/rand_mio_{i}.npy')    
    
    opt_mapping = np.zeros(len(conn))
    opt_mapping[ctx == 1] = np.load(f'{NULL_DIR}/class_mapping_{i}.npy')
    opt_mapping[ctx == 0] = np.max(opt_mapping)+1

    opt_comm = get_communities(opt_mapping)
    
    G = nx.from_numpy_array(conn, parallel_edges=False)
    
    rsn_mod_ = nx_comm.modularity(G, rsn_comm)
    opt_mod_ = nx_comm.modularity(G, opt_comm)
    
    rsn_mod.append(rsn_mod_)
    opt_mod.append(opt_mod_)


np.save(f'{NULL_DIR}/rsn_mod.npy', rsn_mod)
np.save(f'{NULL_DIR}/opt_mod.npy', opt_mod)
  
rsn_mod_null = np.load(f'{NULL_DIR}/rsn_mod.npy')
opt_mod_null = np.load(f'{NULL_DIR}/opt_mod.npy')
    

#%% EMPIRICAL NETWORKS
rsn_comm = get_communities(rsn_mapp_int)

rsn_mod = []
opt_mod = []
for i in range(1000):
    
    print(i)
    
    conn = np.load(f'{CONN_DIR}/consensus_{i}.npy')    
    
    opt_mapping = np.zeros(len(conn))
    opt_mapping[ctx == 1] = np.load(f'{CONN_DIR}/class_mapping_{i}.npy')
    opt_mapping[ctx == 0] = np.max(opt_mapping)+1

    opt_comm = get_communities(opt_mapping)
    
    G = nx.from_numpy_array(conn, parallel_edges=False)
    
    rsn_mod_ = nx_comm.modularity(G, rsn_comm)
    opt_mod_ = nx_comm.modularity(G, opt_comm)
    
    rsn_mod.append(rsn_mod_)
    opt_mod.append(opt_mod_)


np.save(f'{CONN_DIR}/rsn_mod.npy', rsn_mod)
np.save(f'{CONN_DIR}/opt_mod.npy', opt_mod)
 
rsn_mod_emp = np.load(f'{CONN_DIR}/rsn_mod.npy')
opt_mod_emp = np.load(f'{CONN_DIR}/opt_mod.npy')
    
   
#%%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dists(dist_dict, title=None):
    
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot(111)
    
    for label, dist in dist_dict.items():
        sns.distplot(dist,
                     bins=50,
                     hist=False,
                     kde=True,
                     kde_kws={'shade':True},
                     label=label,
                     ax=ax
                     )
    
    
    #ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.get_yaxis().set_visible(False)
    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper left')
#    ax.set_xlim(0.35, 0.65) #-0.05, 0.35)
    
#    if title is not None: plt.suptitle(title)
    
    sns.despine(offset=10, left=True, trim=True)
    fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs', f'dist_{title}.png'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


res_dict = {
            'rsn_mod_emp':rsn_mod_emp,
            'rsn_mod_null':rsn_mod_null,
            'opt_mod_emp':opt_mod_emp,
            'opt_mod_null':opt_mod_null
            }    


plot_dists(res_dict, title='modularity')
           
