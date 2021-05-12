# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 06:49:21 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm

#%%
CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability/scale500'
#NULL_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/significance_mod/scale500'

ctx = np.load('E:/P3_RC/neuromorphic_networks/data/cortical/cortical_human_500.npy')
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

labels, counts = np.unique(rsn_mapp_int, return_counts=True)

spins = np.genfromtxt('E:/P3_RC/neuromorphic_networks/data/spin_test/spin_human_500.csv', delimiter=',').astype(int)[:,:1000]

#%%
def get_communities(class_mapping):
    
    return [np.where(class_mapping == label)[0] for label in np.unique(class_mapping)]

#%%
conn = np.load('E:/P3_RC/neuromorphic_networks/data/connectivity/consensus/human_500.npy')
G = nx.from_numpy_array(conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])], parallel_edges=False)

mod_spin = []
for i in range(1000):
    
    _, counts = np.unique(spins[:,i], return_counts=True)
    print(np.sum(counts))
    
    mapping = rsn_mapp_int[ctx ==1][spins[:,i]]
    cis = get_communities(mapping)
    mod_spin.append(nx_comm.modularity(G, cis))


#%%
mod_reli = []
for i in range(1000):
    conn = np.load(f'{CONN_DIR}/consensus_{i}.npy') 
    G = nx.from_numpy_array(conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])], parallel_edges=False)
    cis = get_communities(rsn_mapp_int[ctx ==1])
    mod_reli.append(nx_comm.modularity(G, cis))
    
    
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
            'mod_spin':mod_spin,
            'mod_reli':mod_reli,
            }    


plot_dists(res_dict, title='modularity')
           
    
  
for i in range(1000):

    a = np.unique(spins[:,i])
    print(len(a))
    
    
    