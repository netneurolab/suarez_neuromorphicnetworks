"""
Created on Mon Jul  6 11:06:24 2020

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd


#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL DYNAMIC VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CLASS = 'functional'
INPUTS = 'subctx' #'thalamus'

#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL STATIC VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

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


def concatenate_tsk_results(path, partition, coding, scores2return, include_alpha, n_samples=1000):

    df_scores = []
    df_avg_scores_per_class = []
    df_avg_scores_per_alpha = []
    for sample_id in range(n_samples):

        print(f'sample_id:  {sample_id}')
        succes_sample = True
        try:
            scores = pd.read_csv(os.path.join(path, f'{partition}_{coding}_score_{sample_id}.csv')).reset_index(drop=True)

        except:
            succes_sample = False
            print('\n Could not find sample No.  ' + str(sample_id))
            pass

        if succes_sample:

            # all scores (per alpha and per class)
            if 'scores' in scores2return:
                scores['coding'] = coding
                scores['sample_id'] = sample_id
                df_scores.append(scores[['sample_id', 'coding', 'class', 'alpha', 'performance', 'capacity', 'n_nodes']])

            # avg scores across alphas per class
            if 'avg_scores_per_class' in scores2return:
                avg_scores_per_class = get_avg_scores_per_class(scores.copy(),
                                                                include_alpha=include_alpha,
                                                                coding=coding
                                                                )

                avg_scores_per_class['coding'] = coding
                avg_scores_per_class['sample_id'] = sample_id
                df_avg_scores_per_class.append(avg_scores_per_class[['sample_id', 'coding', 'class', 'performance', 'capacity', 'n_nodes']])

            # avg scores across classes per alpha
            if 'avg_scores_per_alpha' in scores2return:
                avg_scores_per_alpha = get_avg_scores_per_alpha(scores.copy(),
                                                                include_alpha=None,
                                                                coding=coding
                                                                )

                avg_scores_per_alpha['coding'] = coding
                avg_scores_per_alpha['sample_id'] = sample_id
                df_avg_scores_per_alpha.append(avg_scores_per_alpha[['sample_id', 'coding', 'alpha', 'performance', 'capacity', 'n_nodes']])

    res_dict = {}
    if 'scores' in scores2return:
        df_scores = pd.concat(df_scores).reset_index(drop=True)
        res_dict['scores'] = df_scores

    if 'avg_scores_per_class' in scores2return:
        df_avg_scores_per_class = pd.concat(df_avg_scores_per_class).reset_index(drop=True)
        res_dict['avg_scores_per_class'] = df_avg_scores_per_class

    if 'avg_scores_per_alpha' in scores2return:
        df_avg_scores_per_alpha = pd.concat(df_avg_scores_per_alpha).reset_index(drop=True)
        res_dict['avg_scores_per_alpha'] = df_avg_scores_per_alpha

    return res_dict


def get_avg_scores_per_class(df_scores, include_alpha, coding='encoding'):

    if include_alpha is None: include_alpha = np.unique(df_scores['alpha'])

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    # filter scores by values of alpha
    df_scores = pd.concat([df_scores.loc[np.isclose(df_scores['alpha'], alpha), :] for alpha in include_alpha])

    # average scores across alphas per class
    avg_scores = []
    for clase in class_labels:
        tmp = df_scores.loc[df_scores['class'] == clase, ['performance', 'capacity', 'n_nodes']]\
                       .reset_index(drop=True)
        avg_scores.append(tmp.mean())
    avg_scores = pd.concat(avg_scores, axis=1).T

    # dataFrame with avg coding scores per class
    df_avg_scores = pd.DataFrame(data = np.column_stack((class_labels, avg_scores)),
                                 columns = ['class', 'performance', 'capacity', 'n_nodes'],
                                 ).reset_index(drop=True)

    df_avg_scores['performance']  = df_avg_scores['performance'].astype('float')
    df_avg_scores['capacity']     = df_avg_scores['capacity'].astype('float')
    df_avg_scores['n_nodes']      = df_avg_scores['n_nodes'].astype('float').astype('int')

    return df_avg_scores


def get_avg_scores_per_alpha(df_scores, include_alpha, coding='encoding'):

    if include_alpha is None: include_alpha = np.unique(df_scores['alpha'])

    # average scores across alphas per class
    avg_scores = []
    for alpha in include_alpha:
        tmp = df_scores.loc[np.isclose(df_scores['alpha'], alpha), ['performance', 'capacity', 'n_nodes']]\
                       .reset_index(drop=True)
        avg_scores.append(tmp.mean())
    avg_scores = pd.concat(avg_scores, axis=1).T

    # dataFrame with avg coding scores per class
    df_avg_scores = pd.DataFrame(data = np.column_stack((include_alpha, avg_scores)),
                                 columns = ['alpha', 'performance', 'capacity', 'n_nodes'],
                                ).reset_index(drop=True)

    df_avg_scores['performance']  = df_avg_scores['performance'].astype('float')
    df_avg_scores['capacity']     = df_avg_scores['capacity'].astype('float')
    df_avg_scores['n_nodes']      = df_avg_scores['n_nodes'].astype('float').astype('int')

    return df_avg_scores


#%% --------------------------------------------------------------------------------------------------------------------
def concat_tsk_results(connectome, analysis, dynamics, coding='encoding', n_samples=1000):
    """
        connectome (str): 'human_250', 'human_500'
        analysis (str): 'reliability', 'significance', 'spintest'
        dynamics (str): 'stable', 'edge_chaos', 'chaos'
    """

    output_dir = os.path.join(PROC_RES_DIR, 'tsk_results', analysis, f'{INPUTS}_scale{connectome[-3:]}')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    outputs = {
               'scores':os.path.join(output_dir, f'{CLASS}_{coding}.csv'),
               'avg_scores_per_alpha':os.path.join(output_dir, f'{CLASS}_avg_{coding}.csv'),
               'avg_scores_per_class':os.path.join(output_dir, f'{CLASS}_avg_{coding}_{dynamics}.csv')
              }

#    alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
#    alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

    include_alpha = {
                     'stable':[0.3, 0.5, 0.7, 0.8, 0.9],
                     'edge_chaos':[1.0],
                     'chaos': [1.5, 2.0, 2.5, 3.0, 3.5] # [1.1, 1.2, 1.3, 1.4, 1.5] #
                     }

    scores2return = []
    for output, path in outputs.items():
        if not os.path.exists(path):
            scores2return.append(output)

    input_dir = os.path.join(RAW_RES_DIR, 'tsk_results', analysis, f'{INPUTS}_scale{connectome[-3:]}')

    res = concatenate_tsk_results(path=input_dir,
                                  partition=CLASS,
                                  coding=coding,
                                  scores2return=scores2return,
                                  include_alpha=include_alpha[dynamics],
                                  n_samples=n_samples,
                                  )

    for coding_score, df in res.items():
        df['analysis'] = analysis
        df.to_csv(outputs[coding_score], index=False)


#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    CONNECTOMES = [
                   'human_250',
                   # 'human_500',
                  ]

    ANALYSES   =  {
                   'reliability':10,
                   'significance':10,
                   'spintest':10,
                   }

    DYNAMICS   =  [
                   'stable',
                   'edge_chaos',
                   'chaos',
                  ]

    for connectome in CONNECTOMES[::-1]:
       for analysis, n_samples in ANALYSES.items():
           for dyn_regime in DYNAMICS:
               concat_tsk_results(connectome,
                                  analysis,
                                  dyn_regime,
                                  coding='encoding',
                                  n_samples=n_samples
                                  )

               concat_tsk_results(connectome,
                                  analysis,
                                  dyn_regime,
                                  coding='decoding',
                                  n_samples=n_samples
                                  )
