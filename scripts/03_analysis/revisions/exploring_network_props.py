# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:15:39 2019

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh
from networkx.algorithms import clique
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from netneurotools import modularity
from plotting import plot_tasks

#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'memory_capacity'
CONNECTOME = 'human_500'
CLASS = 'functional'
INPUTS = 'subctx'
ANALYSIS = 'reliability'

COLORS = sns.color_palette("husl", 8)
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])

#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = 'E:/P3_RC/neuromorphic_networks' #os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', TASK, ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')
NET_PROP_DIR = os.path.join(PROC_RES_DIR, 'net_props_results', ANALYSIS, f'scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(coding):
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_{coding}.csv'))
    return avg_scores

def load_net_props():
    df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_local_net_props.csv'), index_col=0)
    return df_net_props

def merge_net_props_n_scores(coding):

    df_net_props = load_net_props()
    net_props = list(df_net_props.columns[2:])

    df_scores = load_avg_scores_per_class(coding)

    # merge network properties and scores
    df = pd.merge(df_scores, df_net_props,
                  on=['sample_id', 'class'],
                  left_index=True,
                  right_index=False
                  ).reset_index(drop=True)

    return df, net_props


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
#df, network_props = merge_net_props_n_scores('encoding')    

#%%
#alpha = 1.0
#df = df.loc[np.isclose(df['alpha'], alpha), :].reset_index(drop=True)
#df = df.rename(columns={'performance': 'MC'})

#%% --------------------------------------------------------------------------------------------------------------------
# CYCLES
# ----------------------------------------------------------------------------------------------------------------------
CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability/scale500'
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_mapp_int = rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

for sample_id in range(1):
    
    conn = np.load(os.path.join(CONN_DIR, f'consensus_{sample_id}.npy')).astype(bool).astype(int)
        
    for rsn in rsn_labels[:1]:
        
        idx = np.where(rsn_mapp == rsn)[0]
        tmp_conn = conn.copy()[np.ix_(idx, idx)]

        G = nx.Graph(tmp_conn)

        a = len(list(nx.cycle_basis(G,1)))
    
    
    
























#%% --------------------------------------------------------------------------------------------------------------------
# MC VS NETWORK PROPERTIES ACROSS SAMPLES
# ----------------------------------------------------------------------------------------------------------------------
#for i, rsn in enumerate(rsn_labels[:7]):
#    
#    tmp = df.loc[df['class'] == rsn, :].reset_index(drop=True)
#    y = tmp['MC']
#    y = (y - min(y))/(max(y) - min(y))
#   
#    sns.set(style="ticks", font_scale=3.0)
#    fig, axs = plt.subplots(1,2, figsize=(18, 8), 
#                            sharex=False, 
#                            sharey=False,
#                            subplot_kw={'xlim':(0, 1),
#                                        'ylim':(0, 1)
#                                        }
#                            )
#
#
##    fig = plt.figure(figsize=(80,8))
#            
#    fig.subplots_adjust(wspace=0.1, left=0.01) #hspace=0.2, 
#    axs = axs.ravel()
#    for j, prop in enumerate(['node_strength', 'wei_clustering_coeff']):
#            
##            ax = plt.subplot(1,10,j+1)
#   
#            x = tmp[prop]
#            x = (x - min(x))/(max(x) - min(x))
#
#            sns.regplot(x=x,
#                        y=y,
#                        color=COLORS[i],
#                        label=f'R = {np.round(np.corrcoef(x,y)[1][0], 2)}',
#                        ax=axs[j]
#                        )
#            if j == 1: axs[j].set_ylabel('') #yaxis.set_visible(False)
#            axs[j].legend()
#            
#    sns.despine(offset=10, trim=True)
#    fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', f'corr_{rsn}.png'), transparent=True, bbox_inches='tight', dpi=300)
#    plt.show()
#    plt.close()
#   

