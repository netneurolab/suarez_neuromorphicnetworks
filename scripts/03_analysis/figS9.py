# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:06:24 2020

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, RuntimeWarning))#,RuntimeWarning])

import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

from plotting import plotting


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500' #'human500' 'human250'
CLASS = 'functional'     #'functional'
INPUTS = 'subctx'        #'subctx' 'thalamus'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_alpha(analysis, coding):
    RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', analysis, f'{INPUTS}_scale{CONNECTOME[-3:]}')
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}.csv'))
    return avg_scores


#%% --------------------------------------------------------------------------------------------------------------------
# PI - AVG SCORE ACROSS CLASSES PER ALPHA VALUE - ALL REGIMES AT ONCE
# ----------------------------------------------------------------------------------------------------------------------
# load data
ANALYSES = [
            'reliability',
            'significance',
            'spintest',
            'reliability_thr',
            'significance_thr',
            'spintest_thr',
            
            ]

df_brain_scores = []
for analysis in ANALYSES:
    avg_scores = load_avg_scores_per_alpha(analysis, 'encoding')
    df_brain_scores.append(avg_scores)

df_brain_scores = pd.concat(df_brain_scores)
df_brain_scores = df_brain_scores.query("sample_id <= 999").reset_index(drop=True)

# scale data
score = 'performance'
min_score = np.min(df_brain_scores[score].values)
max_score = np.max(df_brain_scores[score].values)
df_brain_scores[score] = (df_brain_scores[score]-min_score)/(max_score-min_score)


# boxplot
score = 'performance'
# include_alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

# df_brain_scores = pd.concat([df_brain_scores.loc[np.isclose(df_brain_scores['alpha'], alpha), :] for alpha in include_alpha])\
#                     .reset_index(drop=True)

# boxplot
plotting.boxplot(x='alpha', y=score, df=df_brain_scores.copy(),
                 palette=sns.color_palette('husl', 5),
                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(0,1),
                 legend=True,
                 width=0.8,
                 figsize=(22,8),
                 showfliers=True,
                 )


#%% --------------------------------------------------------------------------------------------------------------------
# PI - AVG SCORE ACROSS CLASSES PER ALPHA VALUE - ALL REGIMES AT ONCE
# ----------------------------------------------------------------------------------------------------------------------
# load data
ANALYSES = [
            'reliability_thr_ext',
            'significance_thr_ext',
            'spintest_thr_ext',
            ]

df_brain_scores = []
for analysis in ANALYSES:
    avg_scores = load_avg_scores_per_alpha(analysis, 'encoding')
    df_brain_scores.append(avg_scores)

df_brain_scores = pd.concat(df_brain_scores)
df_brain_scores = df_brain_scores.query("sample_id <= 999").reset_index(drop=True)

# scale data
score = 'performance'
min_score = np.min(df_brain_scores[score].values)
max_score = np.max(df_brain_scores[score].values)
df_brain_scores[score] = (df_brain_scores[score]-min_score)/(max_score-min_score)

# boxplot
plotting.boxplot(x='alpha', y=score, df=df_brain_scores.copy(),
                 palette=sns.color_palette('husl', 5),
                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(0,1),
                 legend=True,
                 width=0.8,
                 figsize=(22,8),
                 showfliers=True,
                 )


