# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:15:39 2019

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

import fig5_local

from plotting import plot_tasks


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'memory_capacity'
CONNECTOME = 'human_500'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'
ANALYSIS = 'reliability' # 'reliability' 'significance' 'spintest'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', ANALYSIS, f'scale{CONNECTOME[-3:]}')
NET_PROP_DIR = os.path.join(PROC_RES_DIR, 'net_props_results', ANALYSIS, f'scale{CONNECTOME[-3:]}')
RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', TASK, ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# DEFINE CLASSES
# ----------------------------------------------------------------------------------------------------------------------
ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + CONNECTOME + '.npy'))

if CLASS == 'functional':
    filename = CLASS
    class_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
    class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + CONNECTOME + '.npy'))
    class_mapping_ctx = class_mapping[ctx == 1]

elif CLASS == 'cytoarch':
    filename = CLASS
    class_labels = np.array(['PM', 'AC1', 'AC2', 'PSS', 'PS', 'LIM', 'IC', 'subctx'])
    class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + CONNECTOME + '.npy'))
    class_mapping_ctx = class_mapping[ctx == 1]


#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def stack_networks():
    conn_bin = []
    for sample_id in range(1000):

        print('\n ----------------------------sample_id:  ' + str(sample_id))

        if ANALYSIS == 'reliability':   conn_filename = 'consensus_' + str(sample_id) + '.npy'
        elif ANALYSIS == 'significance':  conn_filename = 'rand_mio_' + str(sample_id) + '.npy'

        # load connectivity data
        conn = np.load(os.path.join(RES_CONN_DIR, conn_filename))

        # scale weights [0,1]
        conn = (conn-conn.min())/(conn.max()-conn.min())

        # normalize by spectral radius
        ew, ev = eigh(conn)
        conn = conn/np.max(ew)

        conn_bin.append(conn.copy().astype(bool).astype(int))

    conn_bin = np.dstack(conn_bin)

    return conn_bin


def relative_density(networks):

    n_subj = conn_bin.shape[-1]
    new_props = []
    for clase in class_labels[:-1]:

        within_conn = conn_bin.copy()[np.ix_(class_mapping == clase, class_mapping == clase, np.arange(n_subj))]

        # relative density
        n_nodes_rel = len(within_conn)
        total_conns_rel = int((n_nodes_rel*(n_nodes_rel-1))/2)
        n_conns_within = np.array([np.sum(np.tril(within_conn[:,:,i], -1)) for i in range(n_subj)])
        density_rel = 100*n_conns_within/total_conns_rel

        # create dataframe
        tmp_df = pd.DataFrame(data = density_rel,
                              columns = ['rel_density'],
                              )

        tmp_df['class'] = clase
        tmp_df['sample_id'] = np.arange(len(density_rel))
        new_props.append(tmp_df)

    new_props = pd.concat(new_props).reset_index(drop=True)
    new_props = new_props[['sample_id', 'class', 'rel_density']]

    return new_props



#%%
df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_local_net_props.csv'), index_col=0)
if 'rel_density' not in df_net_props.columns:

    conn_bin = stack_networks()
    df_new_prop = relative_density(conn_bin)

    new_df_net_props = pd.merge(df_net_props, df_new_prop,
                                on=['sample_id', 'class'],
                                left_index=True,
                                right_index=True
                                ).reset_index(drop=True)

    new_df_net_props.to_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_local_net_props.csv'))



#%% --------------------------------------------------------------------------------------------------------------------
# DEPENDENCE ON RELATIVE DENSITY
# ----------------------------------------------------------------------------------------------------------------------
score = 'performance'
dynamics=['stable', 'edge_chaos', 'chaos']
include_props = [
                 'rel_density',
                 ]


corr, net_props = fig5_local.corr_scores_vs_net_props(dynamics=dynamics,
                                                      score=score,
                                                      coding='encoding',
                                                      include_props=include_props,
                                                      correl='pearson'
                                                      )

fig5_local.distplt_corr_net_props_and_scores(corr=corr.copy(),
                                             net_prop_names=include_props,
                                             dynamics=dynamics
                                             )


for prop in include_props[:]:
    fig5_local.scatterplot_net_prop_vs_scores_group(dynamics=dynamics,
                                                    coding='encoding',
                                                    x=prop,
                                                    y=score
                                                    )


#%% --------------------------------------------------------------------------------------------------------------------
# BETWEEN NETWORK COMPARISON
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(dynamics, coding):
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}_{dynamics}.csv'))#, index_col=0)
    return avg_scores


DYNAMICS = ['stable', 'edge_chaos', 'chaos']
score = 'performance' #'capacity', 'performance'
for dyn_regime in DYNAMICS:

    print(f'-------------------------------- {dyn_regime} --------------------------------')

    df_scores = load_avg_scores_per_class(dyn_regime, 'encoding')

    # ------------------
    df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_local_net_props.csv'), index_col=0)#['sample_id', 'class', 'rel_density']
    df = pd.merge(df_scores, df_net_props,
                  on=['sample_id', 'class'],
                  left_index=True,
                  right_index=False
                  ).reset_index(drop=True)

    df['rel_density'] = (df['rel_density']-min(df['rel_density']))/(max(df['rel_density'])-min(df['rel_density']))

    # ------------------
    plot_tasks.bxplt_scores(df.copy(),
                            score,
                            scale=True,
                            minmax=None,
                            norm_score_by=None,
                            title=dyn_regime,
                            width=0.4,
                            figsize=(8,8),
                            )


    # ------------------
    plot_tasks.bxplt_scores(df.copy(),
                            score,
                            scale=True,
                            minmax=None,
                            norm_score_by='rel_density',
                            title=dyn_regime,
                            width=0.4,
                            figsize=(8,8),
                            )
