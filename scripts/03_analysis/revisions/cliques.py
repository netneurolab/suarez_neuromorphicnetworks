# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:01:24 2021

@author: Estefany Suarez
"""

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
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from plotting import plot_tasks


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'pattern_recognition'
CONNECTOME = 'human_500'
CLASS = 'functional'
INPUTS = 'subctx'
ANALYSIS = 'reliability'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')
RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', TASK, ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')
NET_PROP_DIR = os.path.join(PROC_RES_DIR, 'net_props_results', ANALYSIS, f'scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(dynamics, coding):
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}_{dynamics}.csv'))
    return avg_scores

def load_net_props(scale):
    df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_{scale}_cliques.csv'), index_col=0)
    return df_net_props

def merge_net_props_n_scores(dyn_regime, coding, **kwargs):

    df_net_props = load_net_props(**kwargs)
    net_props = list(df_net_props.columns[2:])    

    df_scores = load_avg_scores_per_class(dyn_regime, coding)

    # merge network properties and scores
    df = pd.merge(df_scores, df_net_props,
                  on=['sample_id', 'class'],
                  left_index=True,
                  right_index=False
                  ).reset_index(drop=True)

    return df, net_props


#%% --------------------------------------------------------------------------------------------------------------------
# XXXXX
# ----------------------------------------------------------------------------------------------------------------------
df, net_props = merge_net_props_n_scores(dyn_regime='edge_chaos', coding='encoding', scale='local')

#%%
COLORS = sns.color_palette("husl", 8)
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_mapp_int = rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])

#k = 10
#%%
for k in range(3,10):
    
    sns.set(style="ticks", font_scale=2.0) 
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(111)

    df[f'{k}-clique'] = (df[f'{k}-clique']-df[f'{k}-clique'].min())/(df[f'{k}-clique'].max()-df[f'{k}-clique'].min())

    for i, rsn in enumerate(rsn_labels[:7]):
        
        if k >= 7 and rsn == 'SM':
            continue
            
        tmp = df.loc[df['class'] == rsn, :]
        
#        tmp[f'{k}-clique'] = tmp[f'{k}-clique']/tmp['n_nodes']
        sns.distplot(tmp[f'{k}-clique'], 
                     hist=False, 
                     kde=True,
#                     kde_kws={
##                             'clip':(0,1), 
#                              'shade':True
#                              },
                     label = rsn,
                     color=COLORS[i]
                     )
    
#    plt.legend()
    if k >= 8:
        ax.set_xlim(0.0, 0.5)
        
    plt.suptitle(f'{k}-vertex clique')
    sns.despine(offset=10, trim=True)
    plt.show()
    plt.close()

    
    
    
    


