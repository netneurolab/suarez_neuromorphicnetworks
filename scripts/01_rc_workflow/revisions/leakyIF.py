# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:27:07 2020

@author: Estefany Suarez
"""


import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

import scipy.io as sio
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from reservoir.network import nulls
from reservoir.tasks import (io, coding)
from reservoir.simulator import sim_lnm

from netneurotools import networks

#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'sgnl_recon' #'sgnl_recon' 'pttn_recog'
SPEC_TASK = 'mem_cap' #'mem_cap' 'nonlin_cap' 'fcn_app'
TASK_REF = 'T1'
FACTOR = 0.0001

INPUTS = 'subctx'
CLASS = 'functional' #'functional' 'cytoarch'

N_PROCESS = 40
N_RUNS = 1000

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_metada(connectome, include_subctx=False):

    ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))

    if CLASS == 'functional':
        filename = CLASS
        class_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'])
        class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + connectome + '.npy'))
        class_mapping_ctx = class_mapping[ctx == 1]

    elif CLASS == 'cytoarch':
        filename = CLASS
        class_labels = np.array(['PM', 'AC1', 'AC2', 'PSS', 'PS', 'LIM', 'IC'])
        class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + connectome + '.npy'))
        class_mapping_ctx = class_mapping[ctx == 1]

    if include_subctx:
        return filename, class_labels, class_mapping

    else:
        return filename, class_labels, class_mapping_ctx


def consensus_network(connectome, coords, hemiid, path_res_conn, iter_id=None, sample=None, **kwargs):

    if iter_id is not None: conn_file = f'consensus_{iter_id}.npy'

    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        # load connectivity data
        CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'individual')
        stru_conn = np.load(os.path.join(CONN_DIR, connectome + '.npy'))

        # remove subctx
        # ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))
        # idx_ctx, = np.where(ctx == 1)
        # stru_conn = stru_conn[np.ix_(idx_ctx, idx_ctx)]

        # remove bad subjects
        bad_subj = [7, 12, 43] #SC:7,12,43 #FC:32
        stru_conn = np.delete(stru_conn, bad_subj, axis=2)

        # remove nans
        nan_subjs = np.unique(np.where(np.isnan(stru_conn))[-1])
        stru_conn = np.delete(stru_conn, nan_subjs, axis=2)
        stru_conn_avg = networks.struct_consensus(data=stru_conn.copy()[:,:,sample],
                                                  distance=cdist(coords, coords, metric='euclidean'),
                                                  hemiid=hemiid[:, np.newaxis]
                                                  )

        stru_conn_avg = stru_conn_avg*np.mean(stru_conn, axis=2)

        np.save(os.path.join(path_res_conn, conn_file), stru_conn_avg)


def null_network(model_name, path_res_conn, iter_id=None, **kwargs):

    if iter_id is not None: conn_file = f'{model_name}_{iter_id}' + '.npy'

    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        new_conn = nulls.construct_network_model(type=model_name, **kwargs)

        np.save(os.path.join(path_res_conn, conn_file), new_conn)


def run_workflow(conn_name, connectome, path_res_conn, path_io, path_res_sim, path_res_tsk, \
                 bin=False, input_nodes=None, output_nodes=None, class_labels=None, class_mapp=None, \
                 scores_file=None, iter_id=None, iter_conn=True, iter_io=False, iter_sim=False, \
                 encode=True, decode=True, **kwargs):

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE FILE NAMES
    # ----------------------------------------------------------------------------------------------------------------------

    # define file connectivity data
    if np.logical_and(iter_id is not None, iter_conn):
        conn_file = conn_name + '_' + str(iter_id) + '.npy'
    else: conn_file = conn_name + '.npy'

    # define file I/O data
    if np.logical_and(iter_id is not None, iter_io):
        input_file  = 'inputs_' + str(iter_id) + '.npy'
        output_file = 'outputs_' + str(iter_id) + '.npy'
    else:
        input_file  = 'inputs.npy'
        output_file = 'outputs.npy'

    # define file simulation data (reservoir states)
    if np.logical_and(iter_id is not None, iter_sim):
        res_states_file = 'reservoir_states_' + str(iter_id) + '.npy'
    else: res_states_file  = 'reservoir_states.npy'

    # define file encoding/decoding scores data
    if np.logical_and(iter_id is not None, scores_file is not None):
        encoding_file = scores_file + '_encoding_score_' + str(iter_id) + '.csv'
        decoding_file = scores_file + '_decoding_score_' + str(iter_id) + '.csv'

    elif np.logical_and(iter_id is not None, scores_file is None):
        encoding_file = 'encoding_score_' + str(iter_id) + '.csv'
        decoding_file = 'decoding_score_' + str(iter_id) + '.csv'

    elif np.logical_and(iter_id is None, scores_file is not None):
        encoding_file = scores_file + '_encoding_score.csv'
        decoding_file = scores_file + '_decoding_score.csv'

    else:
        encoding_file = 'encoding_score.csv'
        decoding_file = 'decoding_score.csv'

    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT CONNECTIVITY DATA
    # ----------------------------------------------------------------------------------------------------------------------

    # load connectivity data
    conn = np.load(os.path.join(path_res_conn, conn_file))
    # ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))
    # conn = conn[np.ix_(np.where(ctx == 1)[0], np.where(ctx == 1)[0])]

    # scale weights [0,1]
    if bin: conn = conn.astype(bool).astype(int)
    # else:   conn = (conn-conn.min())/(conn.max()-conn.min())

    # normalize by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # select input nodes
    if input_nodes is None: input_nodes = np.random.choice(n_nodes, 15, replace=False)
    if output_nodes is None: output_nodes = np.setdiff1d(np.arange(n_nodes), input_nodes)
    np.save(os.path.join(path_res_conn, 'input_nodes.npy'), input_nodes)

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(path_io, input_file)):

        io_kwargs = {'time_len':2050,
                     'step_len':20,
                     'bias':0.5,
                     'n_repeats':3
                    }

        inputs, outputs = io.get_io_data(task=TASK,
                                         task_ref=TASK_REF,
                                         n_nodes=n_nodes,
                                         input_nodes=input_nodes,
                                         **io_kwargs
                                        )

        np.save(os.path.join(path_io, input_file), inputs)
        np.save(os.path.join(path_io, output_file), outputs)


    # --------------------------------------------------------------------------------------------------------------------
    # NETWORK SIMULATION - LEAKY RESERVOIR
    # ----------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(path_res_sim, res_states_file)):

        input_train, input_test = np.load(os.path.join(path_io, input_file))


        reservoir_states_train = sim_lnm.run_sim_oger(conn=conn,
                                                      input_nodes=input_nodes,
                                                      inputs=input_train,
                                                      factor=FACTOR,
                                                      task=SPEC_TASK,
                                                      )

        reservoir_states_test  = sim_lnm.run_sim_oger(conn=conn,
                                                      input_nodes=input_nodes,
                                                      inputs=input_test,
                                                      factor=FACTOR,
                                                      task=SPEC_TASK,
                                                      )

        reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(reservoir_states_train, reservoir_states_test)]
        np.save(os.path.join(path_res_sim, res_states_file), reservoir_states, allow_pickle=False)


    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    kwargs_pttn_recog = {'time_lens':30*np.ones(int(0.5*10*80), dtype=int)} # time_lens = len_pattern * np.ones(0.5 * n_patterns * n_repeats)

    reservoir_states = np.load(os.path.join(path_res_sim, res_states_file), allow_pickle=True)
    # reservoir_states = reservoir_states[:, :, :, np.where(ctx == 1)[0]]
    reservoir_states = reservoir_states.squeeze()
    reservoir_states = np.split(reservoir_states, len(reservoir_states), axis=0)
    reservoir_states = [rs.squeeze() for rs in reservoir_states]

    outputs = np.load(os.path.join(path_io, output_file))

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - ENCODERS
    # ----------------------------------------------------------------------------------------------------------------------
    # try:
    if np.logical_and(encode, not os.path.exists(os.path.join(path_res_tsk, encoding_file))):

        print('\nEncoding: ')
        df_encoding = coding.encoder(task=SPEC_TASK,
                                     target=outputs.copy(),
                                     reservoir_states=reservoir_states.copy(),
                                     output_nodes=output_nodes,
                                     **kwargs_pttn_recog
                                     )

        df_encoding.to_csv(os.path.join(path_res_tsk, encoding_file))
    # except:
    #     pass

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - DECODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(decode, not os.path.exists(os.path.join(path_res_tsk, decoding_file))):

            # binarize cortical adjacency matrix
            conn_bin = conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])].astype(bool).astype(int)

            print('\nDecoding: ')
            df_decoding = coding.decoder(task=SPEC_TASK,
                                         target=outputs.copy(),
                                         reservoir_states=reservoir_states.copy(),
                                         output_nodes=output_nodes,
                                         class_labels=class_labels,
                                         class_mapp=class_mapp,
                                         bin_conn=conn_bin,
                                         **kwargs_pttn_recog
                                         )

            df_decoding.to_csv(os.path.join(path_res_tsk, decoding_file))
    except:
        pass

    # delete reservoir states to release memory storage
    if iter_sim: os.remove(os.path.join(path_res_sim, res_states_file))


#%% --------------------------------------------------------------------------------------------------------------------
# LOCAL
# ----------------------------------------------------------------------------------------------------------------------
def leakyIF(connectome):
    """
       Uses the 70 subjs to generate 1000 bootstrapped samples of 40 subjs
       to reconstruct 1000 consensus matrices

       Different consensus connectivity matrix , same I/O signal
    """

    print ('INITIATING PROCESSING TIME - RELIABILITY')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP = 'leakyIF'

    IO_TASK_DIR  = os.path.join(RAW_RES_DIR, 'io_tasks', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP, f'scale{connectome[-3:]}')
    RES_SIM_DIR  = os.path.join(RAW_RES_DIR, 'sim_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')

    if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
    if not os.path.exists(RES_CONN_DIR): os.makedirs(RES_CONN_DIR)
    if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)


    # --------------------------------------------------------------------------------------------------------------------
    # CREATE CONSENSUS MATRICES
    # ----------------------------------------------------------------------------------------------------------------------
    filename, _, _ = load_metada(connectome)

    # construct Erdos-Renyi
    params = []
    for iter_id in range(N_RUNS):
        tmp = {'model_name':'erdos_renyi',
               'path_res_conn':RES_CONN_DIR,
               'iter_id':iter_id,
               'n':1015
                }

        params.append(tmp)

    pool1 = mp.Pool(processes=N_PROCESS)
    res1 = [pool1.apply_async(null_network, (), p) for p in params]
    for r1 in res1: r1.get()
    pool1.close()

    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    params = []
    for iter_id in range(N_RUNS):
        tmp = {'conn_name':'erdos_renyi',
               'connectome':connectome,
               'scores_file':filename,
               'iter_id':iter_id,
               'iter_conn':True,
               'iter_io':False,
               'iter_sim':True,
               'encode':True,
               'decode':False,
               'path_res_conn':RES_CONN_DIR,
               'path_io':IO_TASK_DIR,
               'path_res_sim':RES_SIM_DIR,
               'path_res_tsk':RES_TSK_DIR,
               # 'input_nodes':[223,455],
               # 'input_nodes':[501,1007],
                }

        params.append(tmp)

    pool2 = mp.Pool(processes=N_PROCESS)
    res2 = [pool2.apply_async(run_workflow, (), p) for p in params]
    for r2 in res2: r2.get()
    pool2.close()

    print ('PROCESSING TIME - RELIABILITY')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")


#%% --------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():

    connectome = 'human_500' #human_250  #human_500
    leakyIF(connectome)


if __name__ == '__main__':
    main()
