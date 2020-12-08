# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:29:16 2020

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


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_250'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'
ANALYSIS = 'reliability' # 'significance' 'reproducibility' 'reliability' 'subj_level'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')
NET_PROP_DIR = os.path.join(PROC_RES_DIR, 'net_props_results', ANALYSIS, f'scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(dynamics, coding):
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}_{dynamics}.csv'))
    return avg_scores

def load_net_props():
    df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_global_net_props.csv'), index_col=0).reset_index(drop=True)
    return df_net_props

def merge_net_props_n_scores(df_scores, df_net_props):

    df = pd.merge(df_scores, df_net_props,
                  on=['sample_id'],
                  left_index=True,
                  right_index=True
                  ).reset_index(drop=True)

    return df


#%% --------------------------------------------------------------------------------------------------------------------
# MERGE SCORES AND GLOBAL NETWORK PROPERTIES
# ----------------------------------------------------------------------------------------------------------------------
score = 'performance'
dynamics=['stable', 'edge_chaos', 'chaos']

df = []
for dyn_regime in dynamics:

    print(f'----------------{dyn_regime}-----------------')

    df_scores = load_avg_scores_per_class(dyn_regime, coding='encoding')

    # get avg scores across classes
    mean_scores = []
    for sample_id in range(1000):
        mean_scores.append(df_scores.loc[df_scores.sample_id == sample_id,score].values.mean())

    df_scores = pd.DataFrame(data = np.column_stack((np.arange(len(mean_scores)), mean_scores)),
                             columns = ['sample_id', score],
                             index = None
                             ).reset_index(drop=True)

    df_net_props = load_net_props()

    tmp_df = merge_net_props_n_scores(df_scores, df_net_props).reset_index(drop=True)
    tmp_df['dyn_regime'] = dyn_regime

    df.append(tmp_df)

df = pd.concat(df)


#%%
score = 'performance'
dynamics=['stable', 'edge_chaos', 'chaos']
net_props = df.columns[2:-1]
colors = sns.color_palette(["#2ecc71", "#3498db",  "#9b59b6"])

for i, prop in enumerate(net_props):

    print('\n' )
    print(f'--------------- property:  {prop}')

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(24,8))

    for j, dyn_regime in enumerate(dynamics):

        ax = plt.subplot(1, 3, j+1)

        # filter by dynamical regime
        tmp_df = df.query("dyn_regime == @dyn_regime").reset_index(drop=True)

        # scale values between 0 and 1
        tmp_df[score] = (tmp_df[score]-min(tmp_df[score]))/(max(tmp_df[score])-min(tmp_df[score]))
        tmp_df[prop] = (tmp_df[prop]-min(tmp_df[prop]))/(max(tmp_df[prop])-min(tmp_df[prop]))

        r, pval = spearmanr(tmp_df[prop], tmp_df[score])
        print(f'{dyn_regime}:   Spearmans rho = {r}    pval = {pval}')

        sns.regplot(x=prop, y=score,
                         data=tmp_df,
                         color=colors[j],
                         marker='o',
                         ax=ax,
                         )

        ax.set_aspect("equal")
#        ax.legend(fontsize=15, frameon=True, ncol=1, loc='lower right')
#        ax.get_legend().remove()
        plt.suptitle(prop)
        ax.set_xlim(0,1)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_ylim(0,1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        sns.despine(offset=10, trim=False)

    fig.savefig(fname=os.path.join(f'C:/Users/User/Dropbox/figures_RC/eps/{prop}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
