# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:06:24 2020

@author: Estefany Suarez
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
CONNECTOME = 'human_250'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'  #'subctx' 'thalamus'


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
ANALYSES = ['reliability', 'significance', 'spintest']

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

#%%
# boxplot
score = 'performance'
include_alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

df_brain_scores = pd.concat([df_brain_scores.loc[np.isclose(df_brain_scores['alpha'], alpha), :] for alpha in include_alpha])\
                    .reset_index(drop=True)

plotting.boxplot(x='alpha', y=score, df=df_brain_scores.copy(),
                 palette=sns.color_palette('husl', 5),
#                 suptitle=score,
                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(0,1),
                 legend=True,
                 width=0.8,
                 fig_name=f'brain_vs_nulls_vs_alpha_{CONNECTOME}_{INPUTS}',
                 figsize=(22,8),
                 showfliers=True,
                 )


#%% --------------------------------------------------------------------------------------------------------------------
# PII - VISUAL INSPECTION SCORES DISTRIBUTION
# ----------------------------------------------------------------------------------------------------------------------
ANALYSES = ['reliability', 'significance', 'spintest'] #'significance', 'spintest']

for alpha in [1.0]: # include_alpha:

    print(f'\n---------------------------------------alpha: ... {alpha} ---------------------------------------')
    tmp_df_alpha = df_brain_scores.loc[np.isclose(df_brain_scores['alpha'], alpha), :]

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot(111)

    for analysis in ANALYSES:
       sns.distplot(tmp_df_alpha.loc[tmp_df_alpha.analysis == analysis, score].values,
                    bins=50,
                    hist=False,
                    kde=True,
                    kde_kws={'shade':True},
                    label=analysis
                    )

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.get_yaxis().set_visible(False)
    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    ax.set_xlim(0.60, 1.0)

    sns.despine(offset=10, left=True, trim=True)
    fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figures_RC/eps', f'dist_brain_{analysis}_{CONNECTOME}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


#%% --------------------------------------------------------------------------------------------------------------------
# PIII - STATISTICAL TESTS
# ----------------------------------------------------------------------------------------------------------------------
def cohen_d_2samp(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    # 2 independent sample t test
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


include_alpha = [1.0]
for alpha in include_alpha:

    print(f'\n---------------------------------------alpha: ... {alpha} ---------------------------------------')

    brain = df_brain_scores.loc[(df_brain_scores.analysis == 'reliability') & (np.isclose(df_brain_scores['alpha'], alpha))]
    rewir = df_brain_scores.loc[(df_brain_scores.analysis == 'significance') & (np.isclose(df_brain_scores['alpha'], alpha))]
    spint = df_brain_scores.loc[(df_brain_scores.analysis == 'spintest') & (np.isclose(df_brain_scores['alpha'], alpha))]

    print('Two-sample Wilcoxon-Mann-Whitney rank-sum test:')
    print(f' Brain median: {np.nanmedian(brain.performance.values)}')
    print(f' Rewired median: {np.nanmedian(rewir.performance.values)}')
    print(f' Spintest median: {np.nanmedian(spint.performance.values)}')

    # ----------------------------------------------------------------------------
    # nonparametric Mann-Whitney U test
    # rewired null model
    Urewir, mannu_p_rewir = stats.mannwhitneyu(brain.performance.values[~np.isnan(brain.performance.values)],
                                               rewir.performance.values[~np.isnan(rewir.performance.values)],
                                               alternative='two-sided'
                                               )
    Urewir = Urewir/(1000*1000)
    print(f'\tmannU. pval - rewired:  {mannu_p_rewir}    Effect size: {Urewir}')


    # spintest null model
    Uspint, mannu_p_spint = stats.mannwhitneyu(brain.performance.values[~np.isnan(brain.performance.values)],
                                               spint.performance.values[~np.isnan(spint.performance.values)],
                                               alternative='two-sided'
                                               )
    Uspint = Uspint/(1000*1000)
    print(f'\tmannU. pval - spintest:  {mannu_p_spint}    Effect size: {Uspint}')

    # ----------------------------------------------------------------------------
    # parametric t-test
    # rewired null model
    print('\n')
    print('Two-sample student t-test:')
    print(f' Brain mean: {np.nanmean(brain.performance.values)}')
    print(f' Rewired mean: {np.nanmean(rewir.performance.values)}')
    print(f' Spintest mean: {np.nanmean(spint.performance.values)}')

    _, ttest_p_rewir = stats.ttest_ind(brain.performance.values[~np.isnan(brain.performance.values)],
                                       rewir.performance.values[~np.isnan(rewir.performance.values)],
                                       equal_var=False
                                       )
    eff_size_rewir = cohen_d_2samp(brain.performance.values[~np.isnan(brain.performance.values)], rewir.performance.values[~np.isnan(rewir.performance.values)])
    print(f'\tttest. pval - rewired:  {ttest_p_rewir}       Effect size:{eff_size_rewir}')

    # spintest null model
    _, ttest_p_spint = stats.ttest_ind(brain.performance.values[~np.isnan(brain.performance.values)],
                                       spint.performance.values[~np.isnan(spint.performance.values)],
                                       equal_var=False
                                       )
    eff_size_spint = cohen_d_2samp(brain.performance.values[~np.isnan(brain.performance.values)], spint.performance.values[~np.isnan(spint.performance.values)])
    print(f'\tttest. pval - spintest:  {ttest_p_spint}       Effect size:{eff_size_spint}')
