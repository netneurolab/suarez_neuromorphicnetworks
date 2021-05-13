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
from scipy.spatial import distance

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

from plotting import plotting


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500'
CLASS = 'functional'
INPUTS = 'subctx'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

coords = np.load(os.path.join(DATA_DIR, 'coords', f'coords_{CONNECTOME}.npy'))
dist = distance.cdist(coords, coords, 'euclidean')

#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_avg_scores_per_alpha(analysis, coding):
    RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results', analysis, f'{INPUTS}_scale{CONNECTOME[-3:]}')
    avg_scores = pd.read_csv(os.path.join(RES_TSK_DIR, f'{CLASS}_avg_{coding}.csv'))
    return avg_scores


#%% --------------------------------------------------------------------------------------------------------------------
# PI - ESTIMATING WIRING COST
# ----------------------------------------------------------------------------------------------------------------------
# load data
score = 'performance'
ANALYSES = ['reliability', 'significance']

df_brain_scores = []
for analysis in ANALYSES:

    if analysis == 'reliability':   conn_fname = 'consensus'
    elif analysis == 'significance':  conn_fname = 'rand_mio'

    avg_scores = load_avg_scores_per_alpha(analysis, 'encoding')
    avg_scores['cost'] = 0

    # estimate wiring cost for every sample
    for sample_id in np.unique(avg_scores.sample_id):

        conn_wei = np.load(os.path.join(RAW_RES_DIR, 'conn_results', analysis, f'scale{CONNECTOME[-3:]}', f'{conn_fname}_{sample_id}.npy'))
        conn_bin = conn_wei.copy().astype(bool).astype(int)

        dist_ = (dist.copy()*conn_bin)[np.tril_indices_from(conn_bin, -1)]
        dist_ = dist_[np.nonzero(dist_)]

        wiring_density = (conn_wei*conn_bin)[np.tril_indices_from(conn_wei, -1)]
        wiring_density = wiring_density[np.nonzero(wiring_density)]

        cost = np.dot(dist_, wiring_density)

        avg_scores.loc[avg_scores.sample_id == sample_id, 'cost'] = cost

    df_brain_scores.append(avg_scores)

df_brain_scores = pd.concat(df_brain_scores)

# estimate and scale score to wiring cost ratio
df_brain_scores['score-to-wiring_cost ratio'] = df_brain_scores[score]/df_brain_scores['cost']

min_score = np.min(df_brain_scores['score-to-wiring_cost ratio'].values)
max_score = np.max(df_brain_scores['score-to-wiring_cost ratio'].values)
df_brain_scores['score-to-wiring_cost ratio'] = (df_brain_scores['score-to-wiring_cost ratio']-min_score)/(max_score-min_score)


#%% --------------------------------------------------------------------------------------------------------------------
# PII - SCORE TO WIRING COST RATIO
# ----------------------------------------------------------------------------------------------------------------------
score = 'score-to-wiring_cost ratio'
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
# PIII - STATISTICAL TESTS
# ----------------------------------------------------------------------------------------------------------------------
def cohen_d_2samp(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    # 2 independent sample t test
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


#statistical tests
score = 'score-to-wiring_cost ratio'
include_alpha = [1.0] # [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]
for alpha in include_alpha:

    print(f'\n---------------------------------------alpha: ... {alpha} ---------------------------------------')

    brain = df_brain_scores.loc[(df_brain_scores.analysis == 'reliability') & (np.isclose(df_brain_scores['alpha'], alpha))]
    rewir = df_brain_scores.loc[(df_brain_scores.analysis == 'significance') & (np.isclose(df_brain_scores['alpha'], alpha))]

    print('Two-sample Wilcoxon-Mann-Whitney rank-sum test:')
    print(f' Brain median: {np.nanmedian(brain[score].values)}')
    print(f' Rewired median: {np.nanmedian(rewir[score].values)}')

    # ----------------------------------------------------------------------------
    # nonparametric Mann-Whitney U test
    # empirical vs rewired null model
    Urewir, mannu_p_rewir = stats.mannwhitneyu(brain[score].values[~np.isnan(brain[score].values)],
                                               rewir[score].values[~np.isnan(rewir[score].values)],
                                               alternative='two-sided'
                                               )
    Urewir = Urewir/(1000*1000)
    print(f'\tmannU. pval - rewired:  {mannu_p_rewir}    Effect size: {Urewir}')

    # ----------------------------------------------------------------------------
    # parametric t-test
    # empirical vs rewired null model
    print('\n')
    print('Two-sample student t-test:')
    print(f' Brain mean: {np.nanmean(brain.performance.values)}')
    print(f' Rewired mean: {np.nanmean(rewir.performance.values)}')

    _, ttest_p_rewir = stats.ttest_ind(np.array(brain[score].values[~np.isnan(brain[score].values)]),
                                       np.array(rewir[score].values[~np.isnan(rewir[score].values)]),
                                       equal_var=False
                                       )
    eff_size_rewir = cohen_d_2samp(brain[score].values[~np.isnan(brain[score].values)], rewir[score].values[~np.isnan(rewir[score].values)])
    print(f'\tttest. pval - rewired:  {ttest_p_rewir}       Effect size:{eff_size_rewir}')

    # ----------------------------------------------------------------------------
    # visual inspection
    tmp_df_alpha = df_brain_scores.loc[np.isclose(df_brain_scores['alpha'], alpha), :]

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(13,7))
    ax = plt.subplot(111)

    for analysis in ANALYSES:
       sns.distplot(tmp_df_alpha.loc[tmp_df_alpha.analysis == analysis, score].values,
                    bins=50,
                    hist=False,
                    kde=True,
                    kde_kws={'shade':True},
                    label=analysis
                    )

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.get_yaxis().set_visible(False)
    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    plt.title('memory capacity - to - wiring cost ratio')

    sns.despine(offset=10, left=True, trim=True)
    plt.show()
    plt.close()


#%% --------------------------------------------------------------------------------------------------------------------
# PII - VISUAL INSPECTION CONNECTION LENGTH DISTRIBUTION
# ----------------------------------------------------------------------------------------------------------------------
# connection-length distribution for a single sample
sample_ids = np.unique(avg_scores.sample_id)
sample_id = np.random.choice(sample_ids, 1)[0]

sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(figsize=(8,8)) 
ax = plt.subplot(111)
for analysis in ANALYSES:

    if analysis == 'reliability':   conn_fname = 'consensus'
    elif analysis == 'significance':  conn_fname = 'rand_mio'

    conn_wei = np.load(os.path.join(RAW_RES_DIR, 'conn_results', analysis, f'scale{CONNECTOME[-3:]}', f'{conn_fname}_{sample_id}.npy'))
    conn_bin = conn_wei.copy().astype(bool).astype(int)
    dist_ = (dist.copy()*conn_bin)[np.tril_indices_from(conn_bin, -1)]
    dist_ = dist_[np.nonzero(dist_)]

    sns.distplot(dist_,
                 bins=50,
                 hist=False,
                 kde=True,
                 kde_kws={'shade':True},
                 label=analysis
                 )

ax.xaxis.set_major_locator(MultipleLocator(50))
ax.set_xlim(0, 200)
ax.get_yaxis().set_visible(False)
ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
plt.title('connection length distribution')

sns.despine(offset=10, left=True, trim=True)
plt.show()
plt.close()
