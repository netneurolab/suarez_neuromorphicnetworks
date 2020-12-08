# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['ps.usedistiller'] = 'xpdf'
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import seaborn as sns

from . import plotting


COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'

norm_score_by = None #'n_nodes'

# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def sort_class_labels(class_labels):

    if 'subctx' in class_labels:
        rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx']
    else:
        rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']

    if class_labels.all() in rsn_labels:
        return np.array([clase for clase in rsn_labels if clase in class_labels])

    else:
        return class_labels


def merge_scores(df_scores):

    df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', :] \
                         .rename(columns={'performance':'encoding performance', 'capacity':'encoding capacity'}) \
                         .reset_index(drop=True)

    df_encoding_scores.fillna({'encoding_performance':np.nanmean(df_encoding_scores['encoding performance']), \
                               'encoding capacity':np.nanmean(df_encoding_scores['encoding capacity'])}, \
                                inplace=True)


    df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', :] \
                         .rename(columns={'performance':'decoding performance', 'capacity':'decoding capacity'})\
                         .reset_index(drop=True)

    df_decoding_scores.fillna({'decoding performance':np.nanmean(df_decoding_scores['decoding performance']), \
                               'decoding capacity':np.nanmean(df_decoding_scores['decoding capacity'])}, \
                               inplace=True)


    merge_columns   = list(np.intersect1d(df_encoding_scores.columns, df_decoding_scores.columns))
    df_merge_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=merge_columns, left_index=True, right_index=True).reset_index(drop=True)
    df_merge_scores = df_merge_scores.drop(columns={'coding'})

    # df_merge_scores['coding performance'] = (df_merge_scores['encoding performance'] - df_merge_scores['decoding performance']).astype(float)
    # df_merge_scores['coding capacity']    = (df_merge_scores['encoding capacity']   - df_merge_scores['decoding capacity']).astype(float)

    # df_merge_scores = df_merge_scores[['sample_id', 'class',
    #                                    'encoding performance', 'decoding performance', 'coding performance', \
    #                                    'encoding capacity', 'decoding capacity', 'coding capacity', \
    #                                    'n_nodes', 'analysis']]

    df_merge_scores = df_merge_scores[['sample_id', 'class',
                                       'encoding performance', 'decoding performance', \
                                       'encoding capacity', 'decoding capacity', \
                                       'n_nodes', 'analysis']]

    return df_merge_scores


# --------------------------------------------------------------------------------------------------------------------
# P I - BETWEEN NETWORK COMPARISON AS A FCN OF ALPHA
# ----------------------------------------------------------------------------------------------------------------------
def lnplt_scores_vs_alpha(df, score, include_alpha=None, scale=True, minmax=None, norm_score_by=None, **kwargs):

    if norm_score_by is not None:
        # regress out a variable from coding scores
        x = np.array(df[norm_score_by].values)[:, np.newaxis]
        reg = LinearRegression().fit(x, y=df[score])
        df[score] = df[score] - reg.predict(x)

    if include_alpha is not None:
        df = pd.concat([df.loc[np.isclose(df['alpha'], alpha), :] for alpha in include_alpha])\
               .reset_index(drop=True)

    if scale:
        if minmax is None:
            min_score = np.min(df[score])
            max_score = np.max(df[score])
        else:
            min_score = minmax[0]
            max_score = minmax[1]

        df[score] = (df[score]-min_score)/(max_score-min_score)

    # ----------------------------------------------------
    df = df.rename(columns={'alpha':r'$\alpha$'})
    plotting.lineplot(x=r'$\alpha$', y=score,
                      df=df,
                      palette=COLORS[:-1],
                      hue='class',
                      hue_order=sort_class_labels(np.unique(df['class'])),
                      fig_name='ln_encod_vs_alpha',
                      **kwargs
                      )


# --------------------------------------------------------------------------------------------------------------------
# P II - BETWEEN NETWORK COMPARISON - AVG ACROSS ALPHA
# ----------------------------------------------------------------------------------------------------------------------
def bxplt_scores(df_scores, score, order=None, scale=True, minmax=None, norm_score_by=None, title=None, **kwargs):

    if norm_score_by is not None:
        # regress out a variable from coding scores
        X = np.array(df_scores[norm_score_by])[:, np.newaxis]
        reg_enc = LinearRegression().fit(X, y=df_scores[score])
        df_scores[score] = df_scores[score] - reg_enc.predict(X)

    if scale:
        if minmax is None:
            min_score = np.min(df_scores[score])
            max_score = np.max(df_scores[score])

        else:
            min_score = minmax[0]
            max_score = minmax[1]

        df_scores[score] = ((df_scores[score]-min_score)/(max_score-min_score))

    if order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in order])
    else:
        palette = COLORS[:-1]

    # ----------------------------------------------------
    plotting.boxplot(x='class', y=score,
                     df=df_scores,
                     order=order,
                     palette=palette,
                     legend=True,
                     # title=f'regime: {title}',
                     fig_name=f'bx_enc_{title}',
                     **kwargs
                     )


# --------------------------------------------------------------------------------------------------------------------
# P IV - ENCODING VS DECODING
# ----------------------------------------------------------------------------------------------------------------------
def jointplot_enc_vs_dec(df_scores, score, scale=True, minmax=None, hue_order=None, kind='scatter', draw_line=True, title=None, **kwargs):

    df_scores = merge_scores(df_scores)

    if scale:
        if minmax is None:
            max_score = max(np.max(df_scores[f'decoding {score}']), np.max(df_scores[f'encoding {score}']))
            min_score = min(np.min(df_scores[f'decoding {score}']), np.min(df_scores[f'encoding {score}']))
        else:
            min_score = minmax[0]
            max_score = minmax[1]

        df_scores[f'encoding {score}'] = ((df_scores[f'encoding {score}']-min_score)/(max_score-min_score))
        df_scores[f'decoding {score}'] = ((df_scores[f'decoding {score}']-min_score)/(max_score-min_score))

    # ----------------------------------------------------------------------
    sns.set(style="ticks", font_scale=2.0)

    class_labels = sort_class_labels(np.unique(df_scores['class']))
    if hue_order is None: hue_order = class_labels.copy()
    palette = [COLORS[np.where(class_labels == clase)[0][0]] for clase in hue_order]

    tmp = df_scores.loc[df_scores['class'] == hue_order[0], :]
    g = sns.JointGrid(x=tmp[f'decoding {score}'].values,
                      y=tmp[f'encoding {score}'].values,
                      dropna=True,
                      height=8,
                      ratio=7,
                      **kwargs
                      )

    if kind == 'kde':
        g.plot_joint(sns.kdeplot, color=palette[0], shade=True, shade_lowest=False) # label=class_labels[0], legend=False) #
    elif kind == 'scatter':
        g.plot_joint(sns.scatterplot, color=palette[0], s=50, linewidths=0.5, alpha=0.3, edgecolor='face') # label=class_labels[0], legend=False) #

    g.plot_marginals(sns.distplot, hist=False, kde=True, kde_kws={"shade": True}, color=palette[0])

    for i, clase in enumerate(hue_order[1:]):

        tmp = df_scores.loc[df_scores['class'] == clase, :]
        g.x = tmp[f'decoding {score}'].values
        g.y = tmp[f'encoding {score}'].values

        if kind == 'kde':
            g.plot_joint(sns.kdeplot, color=palette[i+1], shade=True, shade_lowest=False) # label=clase, legend=False) #
        elif kind == 'scatter':
            g.plot_joint(sns.scatterplot, color=palette[i+1], s=50, linewidths=0.5, alpha=0.3, edgecolor='face') # label=clase, legend=False) #

        g.plot_marginals(sns.distplot, hist=False, kde=True, kde_kws={"shade": True}, color=palette[i+1])

    g.ax_joint.set_xlabel(f'decoding {score}')
    g.ax_joint.set_ylabel(f'encoding {score}')

    #g.ax_joint.get_legend().remove()
    #g.ax_joint.legend(fontsize=10, frameon=False, ncol=1, loc='lower right', title='rsn')
    #plt.legend(fontsize=10, frameon=False, ncol=1)#, loc='lower right')

    if draw_line:
        g.x = [0.05,0.95]
        g.y = [0.05,0.95]
        g.plot_joint(sns.lineplot, color='dimgrey', linestyle='--', linewidth=0.8)

    # plt.suptitle(f'regime : {title}', fontsize=25)

    g.savefig(f'C:/Users/User/Dropbox/figures_RC/eps/jnt_enc_vs_dec_{title}.eps')
    plt.show()
    plt.close()


def ttest(df_scores, score, fdr_correction=True):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    pval = []
    tstat = []
    for clase in class_labels:
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'), :][score].values
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'), :][score].values

        t, p = stats.ttest_1samp(encod_scores-decod_scores, popmean=0.0)

        pval.append(p)
        tstat.append(t)

    if fdr_correction: pval = multipletests(pval, 0.05, 'bonferroni')[1]

    return tstat, pval


def effect_size(df_scores, score, minmax=None):

    # ---------------------------------------------------
    def cohen_d_1samp(x, mu=0.0):
        return (np.mean(x) - mu) / np.std(x)

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    effect_size = []
    for clase in class_labels:
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'), :][score].values
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'), :][score].values

        effect_size.append(cohen_d_1samp(encod_scores-decod_scores))

    return np.array(effect_size)


def barplot_eff_size(eff_size, class_labels, title=None):

    df_eff_size = pd.DataFrame(data = np.column_stack((class_labels, eff_size)),
                               columns = ['class', 'effect size'],
                               index = np.arange(len(class_labels))
                               )
    df_eff_size['effect size'] = df_eff_size['effect size'].astype('float')

    # bar plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(8,8))
    ax = plt.subplot(111)

    hue_order = np.array(class_labels)[np.argsort(eff_size)]
    palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])

    sns.barplot(x='class',
                y='effect size',
                data=df_eff_size,
                order=hue_order,
                palette=palette,
                orient='v',
                # width=0.5
                )

    ax.set_ylim(-6,6)
    ax.yaxis.set_major_locator(MultipleLocator(2.0))

    # plt.suptitle(f'regime: {title}', fontsize=25)
    sns.despine(offset=10, trim=True)
    fig.savefig(fname=f'C:/Users/User/Dropbox/figures_RC/eps/bar_effect_size_{title}.eps', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
