# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:18:01 2020

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning,RuntimeWarning))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import seaborn as sns

from plotting import (plotting, plot_tasks)


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_class(analysis, dynamics, coding):
    RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', analysis, f'{INPUTS}_scale{CONNECTOME[-3:]}')
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}_{dynamics}.csv'))
    return avg_scores


#%% --------------------------------------------------------------------------------------------------------------------
# PI - AVG SCORE ACROSS ALPHA PER CLASS - PER REGIME
# ----------------------------------------------------------------------------------------------------------------------
# load data
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'])
ANALYSES = ['reliability', 'significance_unpert']
DYNAMICS = ['stable', 'edge_chaos', 'chaos']

df_rsn_scores = []
for analysis in ANALYSES:
    for dyn_regime in DYNAMICS:

        if analysis == 'significance_unpert':
            for label in rsn_labels:
                scores = load_avg_scores_per_class(f'{analysis}_{label}', dyn_regime, 'encoding')
                scores['dyn_regime'] = dyn_regime
                # scores.drop('analysis')
                scores['analysis'] = 'significance'
                df_rsn_scores.append(scores)

        else:
            scores = load_avg_scores_per_class(analysis, dyn_regime, 'encoding')
            scores['dyn_regime'] = dyn_regime
            df_rsn_scores.append(scores)

df_rsn_scores = pd.concat(df_rsn_scores)
df_rsn_scores = df_rsn_scores.query("sample_id <= 999").reset_index(drop=True)

# scale avg encoding scores
score = 'performance'
min_score = np.min(df_rsn_scores[score].values)
max_score = np.max(df_rsn_scores[score].values)
df_rsn_scores[score] = (df_rsn_scores[score]-min_score)/(max_score-min_score)


#%%
# boxplot
score = 'performance'
DYNAMICS = ['stable', 'edge_chaos', 'chaos']
class_labels = plot_tasks.sort_class_labels(np.unique(df_rsn_scores['class']))

for dyn_regime in DYNAMICS:

    print(f'\n---------------------------------------  {dyn_regime.upper()}  ---------------------------------------')

    df = df_rsn_scores.loc[df_rsn_scores.dyn_regime == dyn_regime, :]

    if dyn_regime == 'chaos':
        ylim = (0.0, 0.5) #(0.2, 0.75)  #
        y_major_loc = 0.1
    else:
        ylim = (0.5, 1.0)
        y_major_loc = 0.1

    plotting.boxplot(x='class', y=f'{score}',
                     df=df,
                     palette=sns.color_palette('husl', 5),
                     suptitle=f'encoding - {dyn_regime}',
                     hue='analysis',
                     order=None,
                     orient='v',
                     width=0.7, #0.8
                     xlim=None,
                     ylim=ylim, #None,
                     y_major_loc=y_major_loc,
                     legend=False,
                     figsize=(20,8), #12,12
                     showfliers=True,
                     )


#%% --------------------------------------------------------------------------------------------------------------------
# PIII - STATISTICAL TESTS
# ----------------------------------------------------------------------------------------------------------------------
def statistical_test(df, score, fdr_corr=True, test_type='nonparametric'):

    def cohen_d_2samp(x,y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2

        # 2 independent sample t test
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    class_labels = plot_tasks.sort_class_labels(np.unique(df['class']))

    pval_rewir = []
    effs_rewir = []
    for clase in class_labels:

        scores = df.loc[df['class'] == clase, :]

        # reliability scores
        brain = scores.loc[scores['analysis'] == 'reliability', score].values
        rewire = scores.loc[scores['analysis'] == 'significance', score].values

        # ----------------------------------------------------------------------------
        # nonparametric Mann-Whitney U test
        if test_type == 'nonparametric':
            # rewired null model
            u, pval = stats.mannwhitneyu(brain,
                                         rewire,
                                         alternative='two-sided'
                                         )
            eff_size = u/(len(brain)*len(rewire))
            if fdr_corr:pval = multipletests(pval, 0.05, 'bonferroni')[1].squeeze()

            pval_rewir.append(pval)
            effs_rewir.append(eff_size)


        # ----------------------------------------------------------------------------
        # parametric t-test
        if test_type == 'parametric':

            # rewired null model
            _, pval = stats.ttest_ind(brain, rewire, equal_var=False)
            eff_size = cohen_d_2samp(brain, rewire)
            if fdr_corr:pval = multipletests(pval, 0.05, 'bonferroni')[1].squeeze()

            pval_rewir.append(pval)
            effs_rewir.append(eff_size)


    pval_rewir = [float(p) for p in pval_rewir]

    return pval_rewir, effs_rewir

#%%
score = 'performance'
DYNAMICS = ['stable', 'edge_chaos', 'chaos']

for dyn_regime in DYNAMICS:

    print(f'\n\n---------------------- {dyn_regime.upper()} --------------------')

    df = df_rsn_scores.loc[df_rsn_scores.dyn_regime == dyn_regime, :]
    enc_pval_rewir, enc_effs_rewir = statistical_test(df.copy(), score)

    # ----------------------
    print('\t--------------------------------------------')
    print("\t\tBrain vs Rewired - avg across alpha")
    print('\t--------------------------------------------')
    print(f'\t\t{class_labels}')

    print('\n\t\tP-vals:')
    print(f'\t\t{np.round(enc_pval_rewir,3)}')

    print('\n\t\tEffect size:')
    print(f'\t\t{np.round(enc_effs_rewir,2)}')
