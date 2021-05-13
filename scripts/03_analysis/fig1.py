# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:56:13 2020

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from plotting import plot_tasks


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500'
CLASS = 'functional'
ANALYSIS   = 'reliability' # 'reliability' 'significance' 'spintest'
INPUTS = 'subctx'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')
RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', ANALYSIS, f'{INPUTS}_scale{CONNECTOME[-3:]}')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_scores(coding):
    coding_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_{coding}.csv'))
    return coding_scores


# %%--------------------------------------------------------------------------------------------------------------------
# PI - BETWEEN NETWORK COMPARISON - CLASS SCORES DISTRIBUTION AS A FCN OF ALPHA
# --------------------------------------------------------------------------------------------------------------------
df_encoding = load_scores('encoding')
score = 'performance'
include_alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

plot_tasks.lnplt_scores_vs_alpha(df_encoding.copy(),
                                 score,
                                 include_alpha=include_alpha,
                                 scale=True,
                                 minmax=None,
                                 ci='sd',
                                 err_style='band',
                                 markers=True,
                                 marker='o',
                                 markersize=12,
                                 linewidth=2,
                                 dashes=False,
                                 x_major_loc=0.2,
                                 ylim=(0,1.1),
                                 xlim=(0.5,1.5),
                                 legend=True,
                                 figsize=(20,8),
                                 )
