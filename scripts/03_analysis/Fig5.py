# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:34:32 2020

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from plotting import plot_tasks


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'
ANALYSIS = 'reliability' # 'reliability' 'significance' 'spintest'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')
RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(dynamics, coding):
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}_{dynamics}.csv'))
    return avg_scores


#%% --------------------------------------------------------------------------------------------------------------------
# P IV - WITHIN NETWORK COMPARISON - AVG ENC VS DEC ACROSS ALPHA PER CLASS - PER REGIME
# ----------------------------------------------------------------------------------------------------------------------
DYNAMICS = ['stable', 'edge_chaos', 'chaos'] # 'edge_chaos', 'edge+chaos'  'chaos'
score = 'performance' #'capacity', 'performance'

for dyn_regime in DYNAMICS:

    print(f'\n-------------------------------- {dyn_regime.upper()} --------------------------------')

    # load data
    df_encoding = load_avg_scores_per_class(dyn_regime, 'encoding')
    df_decoding = load_avg_scores_per_class(dyn_regime, 'decoding')
    df_scores = pd.concat((df_encoding, df_decoding))

    # encoding vs decoding - jointplot
    plot_tasks.jointplot_enc_vs_dec(df_scores.copy(),
                                    score,
                                    scale=True,
#                                    minmax=(1,16),
                                    xlim=(0,1),
                                    ylim=(0,1),
                                    title=dyn_regime,
                                    )

    # information transfer
    class_labels = np.array(plot_tasks.sort_class_labels(np.unique(df_scores['class'])))

    # statistical tests
    t, pval = plot_tasks.ttest(df_scores.copy(),
                               score,
                               )

    eff_size = plot_tasks.effect_size(df_scores.copy(),
                                      score,
                                      )

    print("Intrinsic networks:")
    print(np.array(class_labels)[np.argsort(eff_size)])
    print(f'p-vals: {np.round(np.array(pval)[np.argsort(eff_size)],4)}')
    print(f'Effect sizes: {np.round(eff_size[np.argsort(eff_size)],3)}')

    plot_tasks.barplot_eff_size(eff_size,
                                class_labels,
                                title=dyn_regime
                                )
