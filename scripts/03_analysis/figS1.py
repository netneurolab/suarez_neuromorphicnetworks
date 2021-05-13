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
CONNECTOME = 'human_500' #'human250'
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
ANALYSES = [
            'reliability',      # empirical brain networks - intrinsic network partition 
            'reliability_mod',  # empirical brain networks - network partition that optimizes modularity
            'significance',     # rewired brain networks - intrinsic network partition 
            'significance_mod', # rewired brain networks - network partition that optimizes modularity
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


#%%
# boxplot
score = 'performance'
include_alpha = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

df_brain_scores = pd.concat([df_brain_scores.loc[np.isclose(df_brain_scores['alpha'], alpha), :] for alpha in include_alpha])\
                    .reset_index(drop=True)

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
# PII - VISUAL INSPECTION SCORES DISTRIBUTION
# ----------------------------------------------------------------------------------------------------------------------
include_alpha = [1.0]

for alpha in include_alpha:

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

#    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.get_yaxis().set_visible(False)
    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
#    ax.set_xlim(0.20, 1.0)

    sns.despine(offset=10, left=True, trim=True)
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


def compare_models(df_brain_scores, m1, m2, alpha, test_type='nonparametric'):
    
    x1 = df_brain_scores.loc[(df_brain_scores.analysis == m1) & (np.isclose(df_brain_scores['alpha'], alpha))]
    x2 = df_brain_scores.loc[(df_brain_scores.analysis == m2) & (np.isclose(df_brain_scores['alpha'], alpha))]


    # ----------------------------------------------------------------------------
    # nonparametric Mann-Whitney U test
    if test_type == 'nonparametric':
        
        print('\tTwo-sample Wilcoxon-Mann-Whitney rank-sum test:')
        print(f'\t {m1} median: {np.nanmedian(x1.performance.values)}')
        print(f'\t {m2} median: {np.nanmedian(x2.performance.values)}')

        U, mannu_p = stats.mannwhitneyu(x1.performance.values[~np.isnan(x1.performance.values)],
                                        x2.performance.values[~np.isnan(x2.performance.values)],
                                        alternative='two-sided'
                                        )
        U = U/(1000*1000)
        print(f'\t mannU. pval - {m1} vs {m2}:  {np.round(mannu_p,3)}    Effect size: {np.round(U,2)}')
    
    # ----------------------------------------------------------------------------
    # parametric t-test
    if test_type == 'parametric':
        
        print('\n')
        print('\tTwo-sample student t-test:')
        print(f'\t {m1} mean: {np.nanmean(x1.performance.values)}')
        print(f'\t {m2} mean: {np.nanmean(x2.performance.values)}')

 
        _, ttest_p = stats.ttest_ind(x1.performance.values[~np.isnan(x1.performance.values)],
                                     x2.performance.values[~np.isnan(x2.performance.values)],
                                     equal_var=False
                                     )
        
        eff_size = cohen_d_2samp(x1.performance.values[~np.isnan(x1.performance.values)], 
                                 x2.performance.values[~np.isnan(x2.performance.values)])
        print(f'\t ttest. pval - {m1} vs {m2}:  {np.round(ttest_p,3)}       Effect size:{np.round(eff_size,2)}')


#%%
include_alpha =  [1.0]
for alpha in include_alpha:

    print(f'\n---------------------------------------alpha: ... {alpha} ---------------------------------------')


    print('\nEmpirical brain networks: ')
    print('Intrinsic network partition vs Optimized partition: ')

    compare_models(df_brain_scores, 
                   m1='reliability', 
                   m2='reliability_mod', 
                   alpha=alpha, 
                   test_type='nonparametric')

    
    print('\nRewired brain networks: ')
    print('Intrinsic network partition vs Optimized partition')
    compare_models(df_brain_scores, 
                   m1='significance', 
                   m2='significance_mod', 
                   alpha=alpha, 
                   test_type='nonparametric')


    print('\nEmpirical vs Rewired brain networks - Intrinsic network partition - (Original result)')
    compare_models(df_brain_scores, 
                   m1='reliability', 
                   m2='significance', 
                   alpha=alpha, 
                   test_type='nonparametric')

    print('\nEmpirical vs Rewired brain networks - Optimized partition')
    compare_models(df_brain_scores, 
                   m1='reliability_mod', 
                   m2='significance_mod', 
                   alpha=alpha, 
                   test_type='nonparametric')

    print('\n Empirical (intrinsic network partition) vs Rewired (optimized partition) brain networks')
    compare_models(df_brain_scores, 
                   m1='reliability', 
                   m2='significance_mod', 
                   alpha=alpha, 
                   test_type='nonparametric')

