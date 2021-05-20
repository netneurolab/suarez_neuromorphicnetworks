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

#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL STATIC VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')


#%% --------------------------------------------------------------------------------------------------------------------
# COMPILE LOCAL NETWORK PROPERTIES RESULTS
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


def concatenate_net_props_results(path, class_mapping, filename, n_samples=1000):

    df_net_props = []
    for sample_id in range(n_samples):

        print('\n sample_id:  ' + str(sample_id))
        success_sample = True

        try:
            net_props = pd.read_csv(os.path.join(path, f'{filename}_{sample_id}.csv'), index_col=0)
            net_prop_names = list(net_props.columns)
            net_props['class'] = class_mapping
            net_props = get_avg_net_props_per_class(net_props)
            net_props['sample_id'] = sample_id

        except:
            success_sample = False
            print('\n Could not find sample No.  ' + str(sample_id))

            pass

        if success_sample:  df_net_props.append(net_props)

    # concatenate dataframes
    df_net_props = pd.concat(df_net_props)
    df_net_props = df_net_props.loc[df_net_props['class'] != 'subctx', :]
    df_net_props = df_net_props.reset_index(drop=True)
    df_net_props = df_net_props[['sample_id', 'class'] + net_prop_names]

    return df_net_props


def get_avg_net_props_per_class(df_net_props):
    """
    Returns a DataFrame with the average network properties per class per subject
    """
    # get class labels
    class_labels = sort_class_labels(np.unique(df_net_props['class']))

    if len(class_labels) > 8:
        return df_net_props

    else:

        class_avg_net_props = {clase: df_net_props.loc[df_net_props['class'] == clase, :].mean() for clase in class_labels}
        # class_avg_net_props = {clase: df_net_props.loc[df_net_props['class'] == clase, :].median() for clase in class_labels}
        class_avg_net_props = pd.DataFrame.from_dict(class_avg_net_props, orient='index').reset_index().rename(columns={'index':'class'})
        class_avg_net_props = class_avg_net_props.loc[:,~class_avg_net_props.columns.duplicated()]

        return class_avg_net_props


#%% --------------------------------------------------------------------------------------------------------------------
def concat_net_props_local(connectome, analysis, n_samples=1000):
    """
        connectome (str): 'human_250', 'human_500'
        analysis (str): 'reliability', 'significance', 'spintest'
    """
    if CLASS == 'functional': class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + connectome + '.npy'))
    elif CLASS == 'cytoarch': class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + connectome + '.npy'))

    output_dir = os.path.join(PROC_RES_DIR, 'net_props_results', analysis, f'scale{connectome[-3:]}')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, f'{CLASS}_local_net_props.csv')):

        input_dir = os.path.join(RAW_RES_DIR, 'net_props_local', analysis, f'scale{connectome[-3:]}')
        df_net_props = concatenate_net_props_results(path=input_dir,
                                                     class_mapping=class_mapping,
                                                     filename=f'{CLASS}_net_props',
                                                     n_samples=n_samples
                                                     )

        df_net_props.to_csv(os.path.join(output_dir, f'{CLASS}_local_net_props.csv'))


def tranfer_net_props_global(connectome, analysis):

    input_dir = os.path.join(RAW_RES_DIR, 'net_props_global', analysis, f'scale{connectome[-3:]}')
    output_dir = os.path.join(PROC_RES_DIR, 'net_props_results', analysis, f'scale{connectome[-3:]}')

    if not os.path.exists(os.path.join(output_dir, f'{CLASS}_global_net_props.csv')):
        os.rename(os.path.join(input_dir, f'{CLASS}_net_props.csv'),
                  os.path.join(output_dir, f'{CLASS}_global_net_props.csv')
                  )


def tranfer_net_props_modular(connectome, analysis):

    input_dir = os.path.join(RAW_RES_DIR, 'net_props_mod', analysis, f'scale{connectome[-3:]}')
    output_dir = os.path.join(PROC_RES_DIR, 'net_props_results', analysis, f'scale{connectome[-3:]}')

    if not os.path.exists(os.path.join(output_dir, f'{CLASS}_modular_net_props.csv')):
        os.rename(os.path.join(input_dir, f'{CLASS}_net_props.csv'),
                  os.path.join(output_dir, f'{CLASS}_modular_net_props.csv')
                  )


def concat_cliques(connectome, analysis, scale='local', n_samples=1000):

    if scale == 'local':

        if CLASS == 'functional': class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + connectome + '.npy'))
        elif CLASS == 'cytoarch': class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + connectome + '.npy'))

        output_dir = os.path.join(PROC_RES_DIR, 'net_props_results', analysis, f'scale{connectome[-3:]}')
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        if not os.path.exists(os.path.join(output_dir, f'{CLASS}_local_cliques.csv')):

            input_dir = os.path.join(RAW_RES_DIR, 'net_props_local', analysis, f'scale{connectome[-3:]}')
            df_net_props = concatenate_net_props_results(path=input_dir,
                                                         class_mapping=class_mapping,
                                                         filename=f'cliques',
                                                         n_samples=n_samples
                                                         )

            df_net_props.fillna(0, inplace=True, downcast='infer')
            df_net_props.to_csv(os.path.join(output_dir, f'{CLASS}_local_cliques.csv'))

    elif scale == 'modular':

        input_dir = os.path.join(RAW_RES_DIR, 'net_props_mod', analysis, f'scale{connectome[-3:]}')
        output_dir = os.path.join(PROC_RES_DIR, 'net_props_results', analysis, f'scale{connectome[-3:]}')

        if not os.path.exists(os.path.join(output_dir, f'{CLASS}_modular_cliques.csv')):
            os.rename(os.path.join(input_dir, f'{CLASS}_cliques.csv'),
                      os.path.join(output_dir, f'{CLASS}_modular_cliques.csv')
                      )



#%% ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    CONNECTOMES = [
                   'human_250',
                   'human_500',
                   ]

    ANALYSES   =  {'reliability':1000}

    for connectome in CONNECTOMES[::-1]:
       for analysis, n_samples in ANALYSES.items():
              concat_net_props_local(connectome, analysis, scale='', n_samples=n_samples)
              tranfer_net_props_global(connectome, analysis)
              tranfer_net_props_modular(connectome, analysis)
              # concat_cliques(connectome, analysis, scale='local')
              # concat_cliques(connectome, analysis, scale='modular', n_samples=1000)
