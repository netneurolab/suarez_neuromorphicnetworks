# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:15:39 2019

@author: Estefany Suarez
"""

import os

os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing as mp
import time

from reservoir.network import network_properties


#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
ANALYSIS = 'reliability' # 'subj_level' 'significance' 'reproducibility' 'reliability'
TASK = 'sgnl_recon' #'sgnl_recon' 'pttn_recog'
SPEC_TASK = 'mem_cap' #'mem_cap' 'nonlin_cap' 'fcn_app'
CONNECTOME = 'human_500'

CLASS = 'functional' #'functional' 'cytoarch'

N_PROCESS = 10
N_RUNS = 1000

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
RES_TSK_DIR = os.path.join(RAW_RES_DIR, 'tsk_results', ANALYSIS, 'scale' + CONNECTOME[-3:])
RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', ANALYSIS, 'scale' + CONNECTOME[-3:])

LOCAL_NET_PROP_DIR = os.path.join(RAW_RES_DIR, 'net_props_local', ANALYSIS, 'scale' + CONNECTOME[-3:])
if not os.path.exists(os.path.join(LOCAL_NET_PROP_DIR)): os.makedirs(LOCAL_NET_PROP_DIR)

GLOBAL_NET_PROP_DIR = os.path.join(RAW_RES_DIR, 'net_props_global', ANALYSIS, 'scale' + CONNECTOME[-3:])
if not os.path.exists(os.path.join(GLOBAL_NET_PROP_DIR)): os.makedirs(GLOBAL_NET_PROP_DIR)


#%% --------------------------------------------------------------------------------------------------------------------
# DEFINE CLASSES
# ----------------------------------------------------------------------------------------------------------------------
ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + CONNECTOME + '.npy'))

if CLASS == 'functional':
    filename = CLASS
    class_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
    class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + CONNECTOME + '.npy'))
    class_mapping_ctx = class_mapping[ctx == 1]

elif CLASS == 'cytoarch':
    filename = CLASS
    class_labels = np.array(['PM', 'AC1', 'AC2', 'PSS', 'PS', 'LIM', 'IC', 'subctx'])
    class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + CONNECTOME + '.npy'))
    class_mapping_ctx = class_mapping[ctx == 1]


#%% --------------------------------------------------------------------------------------------------------------------
# MULTIPLE SAMPLE ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------
def local_network_properties(sample_id):

    if not os.path.exists(os.path.join(LOCAL_NET_PROP_DIR, f'{CLASS}_net_props_' + str(sample_id) + '.csv')):
        print('\n sample_id:  ' + str(sample_id))

        # -------------------------------------------------------------------------------------------------------------------
        # IMPORT CONNECTIVITY DATA
        # -------------------------------------------------------------------------------------------------------------------

        if ANALYSIS == 'reliability':   conn_filename = 'consensus_' + str(sample_id) + '.npy'
        elif ANALYSIS == 'significance':  conn_filename = 'rand_mio_' + str(sample_id) + '.npy'

        # load connectivity data
        conn_wei = np.load(os.path.join(RES_CONN_DIR, conn_filename))

        # scale weights [0,1]
        conn_wei = (conn_wei-conn_wei.min())/(conn_wei.max()-conn_wei.min())

        # normalize by spectral radius
        ew, ev = eigh(conn_wei)
        conn_wei = conn_wei/np.max(ew)


        # --------------------------------------------------------------------------------------------------------------------
        # ESTIMATE NETWORK PROPERTIES
        # ----------------------------------------------------------------------------------------------------------------------

        # estimate network properties
        df_net_props = network_properties.get_local_network_properties(conn=conn_wei,
                                                                 cortical=ctx,
                                                                 class_mapping=class_mapping,
                                                                 include_subctx=True,
                                                                 )

        df_net_props.to_csv(os.path.join(LOCAL_NET_PROP_DIR, f'{CLASS}_net_props_' + str(sample_id) + '.csv'))


def global_network_properties(n_samples):

    if not os.path.exists(os.path.join(GLOBAL_NET_PROP_DIR, f'{CLASS}_net_props.csv')):
        df_net_props = []
        for sample_id in range(n_samples):

            print(f'\n ----------------------------sample_id:    {sample_id}')

            # --------------------------------------------------------------------------------------------------------------------
            # CONNECTIVITY PROFILE
            # ----------------------------------------------------------------------------------------------------------------------
            if ANALYSIS == 'reliability':   conn_filename = 'consensus_' + str(sample_id) + '.npy'
            elif ANALYSIS == 'significance':  conn_filename = 'rand_mio_' + str(sample_id) + '.npy'

            # load connectivity data
            conn = np.load(os.path.join(RES_CONN_DIR, conn_filename))

            # scale weights [0,1]
            conn = (conn-conn.min())/(conn.max()-conn.min())

            # normalize by spectral radius
            ew, ev = eigh(conn)
            conn = conn/np.max(ew)

            # estimate global network properties
            class_mapping_int = np.array([np.where(class_labels == mapp)[0][0] for mapp in class_mapping]).astype(int)
            properties, prop_list = network_properties.get_global_network_properties(conn,
                                                                                     cortical=ctx,
                                                                                     class_mapping=class_mapping_int,
                                                                                     )
            df_net_props.append(np.array(properties)[np.newaxis, :])


        df_net_props = pd.DataFrame(np.vstack(df_net_props),
                                    columns=prop_list).reset_index(drop=True)
        df_net_props['sample_id'] = np.arange(len(df_net_props))
        df_net_props.to_csv(os.path.join(GLOBAL_NET_PROP_DIR, f'{CLASS}_net_props.csv'))


def main():

    # --------------------------------------------------------------------------------------------------------------------
    # ESTIMATE LOCAL NETWORK PROPERTIES
    # ----------------------------------------------------------------------------------------------------------------------
    print ('INITIATING PROCESSING TIME - LOCAL NETWORK PROPERTIES')
    t0_1 = time.clock()
    t0_2 = time.time()

    params = [{'sample_id':sample_id} for sample_id in range(N_RUNS)]

    pool = mp.Pool(processes=N_PROCESS)
    res = [pool.apply_async(local_network_properties, (), p) for p in params]
    for r in res: r.get()

    print ('PROCESSING TIME - LOCAL NETWORK PROPERTIES')
    print (time.clock()-t0_1, "seconds process time")
    print (time.time()-t0_2, "seconds wall time")


    # --------------------------------------------------------------------------------------------------------------------
    # ESTIMATE GLOBAL NETWORK PROPERTIES
    # ----------------------------------------------------------------------------------------------------------------------
    print ('INITIATING PROCESSING TIME - GLOBAL NETWORK PROPERTIES')
    t0_1 = time.clock()
    t0_2 = time.time()

    global_network_properties(n_samples=N_RUNS)

    print ('PROCESSING TIME - GLOBAL NETWORK PROPERTIES')
    print (time.clock()-t0_1, "seconds process time")
    print (time.time()-t0_2, "seconds wall time")


if __name__ == '__main__':
    main()
