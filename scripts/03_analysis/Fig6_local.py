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

def load_net_props():
    df_net_props = pd.read_csv(os.path.join(NET_PROP_DIR, f'{CLASS}_local_net_props.csv'), index_col=0)
    return df_net_props

def merge_net_props_n_scores(dyn_regime, coding):

    df_net_props = load_net_props()
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
# PI - DISTRIBUTION OF CORRELATIONS (SCORES VS NETWORK PROPERTIES) ACROSS REGIMES PER PROPERTY
# ----------------------------------------------------------------------------------------------------------------------
def corr_scores_vs_net_props(dynamics, score, coding, include_props=None, correl=None):

    correl = []
    for dyn_regime in dynamics:

        print(f'\n----------------{dyn_regime}-----------------')

        df, net_props = merge_net_props_n_scores(dyn_regime, coding)
        if include_props is None: include_props = net_props

        samples = np.unique(df['sample_id'])

        # estimate correlation
        tmp_correl = np.zeros((len(samples), len(include_props)))
        for i, sample_id in enumerate(samples):
            tmp_df = df.query("sample_id == @sample_id")

            if correl == 'pearson':
                tmp_correl[i,:] = [np.corrcoef(tmp_df[score].values, tmp_df[prop].values)[0][1] for prop in include_props]
            else:
                tmp_correl[i,:] = [spearmanr(tmp_df[score].values, tmp_df[prop].values)[0] for prop in include_props]

        correl.append(tmp_correl)

    return np.dstack(correl), include_props


def distplt_corr_net_props_and_scores(corr, net_prop_names, dynamics):

    colors = sns.color_palette(["#2ecc71", "#3498db",  "#9b59b6"])

    for j, prop in enumerate(net_prop_names):

        sns.set(style="ticks", font_scale=2.0)
        fig = plt.figure(figsize=(10,7))
        ax = plt.subplot(111)

        for i, dyn_regime in enumerate(dynamics):
            sns.distplot(a=corr[:,j,i],
                         bins=50,
                         hist=False,
                         kde=True,
                         kde_kws={'shade':True, 'clip':(-1,1)},
                         color=colors[i],
                         label=' '.join(dyn_regime.split('_')),
                         )

        ax.set_xlim(-1,1)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        # ax.set_ylim(0,15)
        ax.get_yaxis().set_visible(False)

        # ax.get_legend().remove()

        plt.suptitle(' '.join(prop.split('_')))

        sns.despine(offset=10, trim=True, left=True)
#        fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figures_RC/eps', f'{prop}.eps'), transparent=True, bbox_inches='tight', dpi=300)
        plt.show()


def scatterplot_net_prop_vs_scores_group(dynamics, coding, x, y):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(24, 8))
    COLORS = sns.color_palette("husl", 8)[:-1]

    for i, dyn_regime in enumerate(dynamics):

        ax = plt.subplot(1, len(dynamics), i+1)

        df, _ = merge_net_props_n_scores(dyn_regime, coding)
        class_labels = plot_tasks.sort_class_labels(np.unique(df['class']))

        # scale values
        df[x] = (df[x] - min(df[x]))/(max(df[x]) - min(df[x]))
        df[y] = (df[y] - min(df[y]))/(max(df[y]) - min(df[y]))

        # average values for markers
        av_x = [df.query("`class` == @clase")[x].mean() for clase in class_labels]
        av_y = [df.query("`class` == @clase")[y].mean() for clase in class_labels]

        # standard deviation values for error bars
        sd_x = [df.query("`class` == @clase")[x].std() for clase in class_labels]
        sd_y = [df.query("`class` == @clase")[y].std() for clase in class_labels]

        tmp_df = pd.DataFrame(data = np.column_stack((av_x, sd_x, av_y, sd_y)),
                              columns = [f'avg {x}', f'sd {x}', f'avg {y}',  f'sd {y}'],
                              )

        tmp_df['class'] = class_labels

        # scatter plot
        for j, clase in enumerate(class_labels):

            plt.errorbar(x=tmp_df.query("`class` == @clase")[f'avg {x}'],
                         y=tmp_df.query("`class` == @clase")[f'avg {y}'],
                         xerr=tmp_df.query("`class` == @clase")[f'sd {x}'],
                         yerr=tmp_df.query("`class` == @clase")[f'sd {y}'],
                         ecolor=COLORS[j]
                         )

            plt.scatter(x=tmp_df.query("`class` == @clase")[f'avg {x}'],
                        y=tmp_df.query("`class` == @clase")[f'avg {y}'],
                        s=200,
                        marker='D',
                        edgecolor=COLORS[j],
                        color=COLORS[j],
                        alpha=0.5
                        )

        if i == 1: plt.xlabel(f'avg {x}')
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        plt.xlim(0,1)

        if i == 0: plt.ylabel(f'avg {y}')
        if i == 1 or i == 2: ax.get_yaxis().set_ticklabels([])
        plt.ylim(0,1)

        plt.title(dyn_regime)

    sns.despine(offset=10, trim=False)
#    fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figures_RC/eps', f'{x}_{ANALYSIS}_scttplt.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()




#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    score = 'performance'
    dynamics=['stable', 'edge_chaos', 'chaos']
    include_props = [
                     'node_strength',
    #                 'node_degree',
                     'wei_clustering_coeff',
    #                 'bin_clustering_coeff',
                     'wei_centrality',
    #                 'bin_centrality',
                     'wei_participation_coeff',
    #                 'bin_participation_coeff',
                     'wei_diversity_coeff',
                     ]

    corr, net_props = corr_scores_vs_net_props(dynamics=dynamics,
                                               score=score,
                                               coding='encoding',
                                               include_props=include_props
                                               )

    distplt_corr_net_props_and_scores(corr=corr.copy(),
                                      net_prop_names=include_props,
                                      dynamics=dynamics
                                      )


    for prop in include_props[:]:
        scatterplot_net_prop_vs_scores_group(dynamics=dynamics,
                                             coding='encoding',
                                             x=prop,
                                             y=score
                                             )
