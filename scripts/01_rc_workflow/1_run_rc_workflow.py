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
from reservoir.tasks import (io, coding, tasks)
from reservoir.simulator import sim_lnm

from netneurotools import networks

#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'sgnl_recon' #'sgnl_recon' 'pttn_recog'
SPEC_TASK = 'mem_cap' #'mem_cap' 'nonlin_cap' 'fcn_app'
TASK_REF = 'T1'
FACTOR = 0.0001 #0.0001 0.001 0.01

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

        new_conn = nulls.construct_null_model(type=model_name, **kwargs)

        np.save(os.path.join(path_res_conn, conn_file), new_conn)


def run_workflow(conn_name, connectome, path_res_conn, path_io, path_res_sim, path_res_tsk, \
                 bin=False, input_nodes=None, output_nodes=None, readout_modules=None, \
                 scores_file=None, iter_id=None, iter_conn=True, iter_io=False, iter_sim=False, \
                 encode=True, decode=True, **kwargs):
    """
        Runs the full reservoir pipeline: loads and scales connectivity matrix,
        generates input/output data for the task, simulates reservoir states,
        and trains the readout module.

        Parameters
        ----------
        conn_name: str, {'consensus', 'rand_mio'}
            Specifies the name of the connectivity matrix file. 'consensus' for
            reliability and spintest analyses, and 'rand_mio' for significance
            analysis.

        connectome: str, {human_500, human_250}
            Specifies the scale of the conenctome

        path_res_conn : str
            Path to conenctivity matrix

        path_io : str
            Path to simulation results

        path_res_sim : str
            Path to simulation results

        path_res_tsk : str
            Path to task scores

        bin : bool
            If True, the binary matrix will be used

        input,output nodes: (N,) list or numpy.darray
            List or array that indicates the indexes of the input and output
            nodes in the recurrent network.
            N: number of input,output nodes in the network

        readout_modules: (N, ) numpy.darray
            Array that indicates the module at which each output node belongs
            to. Modules can be int or str.
            N: number of output nodes

        scores_file : str, {'functional', 'cytoarch'}
            Name of the partition used

        iter_id : int
            Number/name of the iteration

        iter_{conn,io,sim} : bool
            If True, specific instances (i.e., connectivity, input/output data,
            network states) related to the iteration indicated by iter_id will
            be used.

        encode,decode : bool
            If True, encoding,decoding will run
    """

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
    ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))

    # scale weights [0,1]
    if bin: conn = conn.astype(bool).astype(int)
    else:   conn = (conn-conn.min())/(conn.max()-conn.min())

    # normalize by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # select input nodes
    if input_nodes is None: input_nodes = np.where(ctx == 0)[0]
    if output_nodes is None: output_nodes = np.where(ctx == 1)[0]

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(path_io, input_file)):

        io_kwargs = {'time_len':2050}

        inputs, outputs = io.get_io_data(task=TASK,
                                         task_ref=TASK_REF,
                                         **io_kwargs
                                        )

        np.save(os.path.join(path_io, input_file), inputs)
        np.save(os.path.join(path_io, output_file), outputs)


    # --------------------------------------------------------------------------------------------------------------------
    # NETWORK SIMULATION - LINEAR MODEL
    # ----------------------------------------------------------------------------------------------------------------------
    alphas = tasks.get_default_alpha_values(SPEC_TASK)
    if not os.path.exists(os.path.join(path_res_sim, res_states_file)):

        input_train, input_test = np.load(os.path.join(path_io, input_file))

        # create input connectivity matrix - depends on the shape of the input
        w_in = np.zeros((input_train.shape[1],len(conn)))
        w_in[input_nodes,:] = FACTOR

        reservoir_states_train = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_train,
                                                 alphas=alphas,
                                                )

        reservoir_states_test  = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_test,
                                                 alphas=alphas,
                                                )

        reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(reservoir_states_train, reservoir_states_test)]
        np.save(os.path.join(path_res_sim, res_states_file), reservoir_states, allow_pickle=False)


    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    reservoir_states = np.load(os.path.join(path_res_sim, res_states_file), allow_pickle=True)
    reservoir_states = reservoir_states[:, :, :, output_nodes]
    reservoir_states = reservoir_states.squeeze()
    reservoir_states = np.split(reservoir_states, len(reservoir_states), axis=0)
    reservoir_states = [rs.squeeze() for rs in reservoir_states]

    outputs = np.load(os.path.join(path_io, output_file))

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - ENCODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(encode, not os.path.exists(os.path.join(path_res_tsk, encoding_file))):

            print('\nEncoding: ')
            df_encoding = coding.encoder(task=SPEC_TASK,
                                         target=outputs,
                                         reservoir_states=reservoir_states,
                                         readout_modules=readout_modules,
                                         alphas=alphas,
                                         )

            df_encoding = df_encoding.rename(columns={'module':'class'}, copy=False)
            df_encoding.to_csv(os.path.join(path_res_tsk, encoding_file))
    except:
        pass

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - DECODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(decode, not os.path.exists(os.path.join(path_res_tsk, decoding_file))):

            # binarize cortical adjacency matrix
            conn_bin = conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])].astype(bool).astype(int)

            print('\nDecoding: ')
            df_decoding = coding.decoder(task=SPEC_TASK,
                                         target=outputs,
                                         reservoir_states=reservoir_states,
                                         readout_modules=readout_modules,
                                         bin_conn=conn_bin,
                                         alphas=alphas,
                                         )

            df_decoding = df_decoding.rename(columns={'module':'class'}, copy=False)
            df_decoding.to_csv(os.path.join(path_res_tsk, decoding_file))

    except:
        pass

    # delete reservoir states to release memory storage
    if iter_sim: os.remove(os.path.join(path_res_sim, res_states_file))


#%% --------------------------------------------------------------------------------------------------------------------
# LOCAL
# ----------------------------------------------------------------------------------------------------------------------
def reliability(connectome):
    """
       Uses the 70 subjs to generate 1000 bootstrapped samples of 40 subjs
       to reconstruct 1000 consensus matrices

       Different consensus connectivity matrix , same I/O signal
    """

    print ('INITIATING PROCESSING TIME - RELIABILITY')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP = 'reliability'

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
    CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'individual')

    filename, class_labels, class_mapping_ctx = load_metada(connectome)

    # generate bootstrapped samples
    if not os.path.exists(os.path.join(RES_CONN_DIR, 'subj_resampling.npy')):

        # load connectivity data
        stru_conn = np.load(os.path.join(CONN_DIR, connectome + '.npy'))

        # remove bad subjects
        bad_subj = [7, 12, 43] #SC:7,12,43 #FC:32
        stru_conn = np.delete(stru_conn, bad_subj, axis=2)

        # perform bootstrap resampling
        n_subj = stru_conn.shape[2]
        resampling = [np.random.choice(np.arange(n_subj), size=40, replace=False) for _ in range(N_RUNS)]
        np.save(os.path.join(RES_CONN_DIR, 'subj_resampling.npy'), resampling)

    else:
        resampling = np.load(os.path.join(RES_CONN_DIR, 'subj_resampling.npy'))

    # load coordinates and hemisphere ids
    coords = np.load(os.path.join(DATA_DIR, 'coords', 'coords_' + connectome + '.npy'))
    hemiid = np.load(os.path.join(DATA_DIR, 'hemispheres', 'hemiid_' + connectome + '.npy'))

    # construct group-consensus SC for bootstrapped samples
    params = []
    for sample_id, sample in enumerate(resampling):
        tmp_params_dict = {'connectome':connectome,
                           'iter_id':sample_id,
                           'coords':coords,
                           'hemiid':hemiid,
                           'path_res_conn':RES_CONN_DIR,
                           'sample':sample,
                           }

        params.append(tmp_params_dict)

    pool1 = mp.Pool(processes=N_PROCESS)
    res1 = [pool1.apply_async(consensus_network, (), p) for p in params]
    for r1 in res1: r1.get()
    pool1.close()

    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    params = []
    for iter_id in range(N_RUNS):

        tmp = {'conn_name':'consensus',
               'connectome':connectome,
               'scores_file':filename,
               'iter_id':iter_id,
               'iter_conn':True,
               'iter_io':False,
               'iter_sim':True,
               'encode':True,
               'decode':False,
               'readout_modules':class_mapping_ctx,
               'path_res_conn':RES_CONN_DIR,
               'path_io':IO_TASK_DIR,
               'path_res_sim':RES_SIM_DIR,
               'path_res_tsk':RES_TSK_DIR,
                }

        params.append(tmp)

    pool2 = mp.Pool(processes=N_PROCESS)
    res2 = [pool2.apply_async(run_workflow, (), p) for p in params]
    for r2 in res2: r2.get()
    pool2.close()

    print ('PROCESSING TIME - RELIABILITY')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")


def significance(connectome):
    """
        Different null connectivity matrix, same I/O signals
    """

    print ('INITIATING PROCESSING TIME - SIGNIFICANCE')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP = 'significance'

    IO_TASK_DIR  = os.path.join(RAW_RES_DIR, 'io_tasks', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP, f'scale{connectome[-3:]}')
    RES_SIM_DIR  = os.path.join(RAW_RES_DIR, 'sim_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')

    if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
    if not os.path.exists(RES_CONN_DIR): os.makedirs(RES_CONN_DIR)
    if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)


    # --------------------------------------------------------------------------------------------------------------------
    # CREATE NULL NETWORK MODELS
    # ----------------------------------------------------------------------------------------------------------------------
    CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'consensus')
    conn_wei = np.load(os.path.join(CONN_DIR, connectome + '.npy'))

    filename, class_labels, class_mapping_ctx = load_metada(connectome)

    params = []
    for iter_id in range(N_RUNS):

        tmp = {'conn':conn_wei.copy(),
               'model_name':'rand_mio',
               'path_res_conn':RES_CONN_DIR,
               'iter_id':iter_id,
               'swaps':10
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

        tmp = {'conn_name':'rand_mio',
               'connectome':connectome,
               'scores_file':filename,
               'iter_id':iter_id,
               'iter_conn':True,
               'iter_io':False,
               'iter_sim':True,
               'encode':True,
               'decode':False,
               'readout_modules':class_mapping_ctx,
               'path_res_conn':RES_CONN_DIR,
               'path_io':IO_TASK_DIR,
               'path_res_sim':RES_SIM_DIR,
               'path_res_tsk':RES_TSK_DIR,
               }

        params.append(tmp)

    pool2 = mp.Pool(processes=N_PROCESS)
    res2 = [pool2.apply_async(run_workflow, (), p) for p in params]
    for r2 in res2: r2.get()
    pool2.close()

    print ('PROCESSING TIME - SIGNIFICANCE')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")


def spintest(connectome):
    """
        Same connectivity matrix, same I/O signals, same reservoir states,
        I/O node assignments based on spintest
    """

    print ('INITIATING PROCESSING TIME - SPINTEST')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP = 'spintest'

    IO_TASK_DIR  = os.path.join(RAW_RES_DIR, 'io_tasks', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_SIM_DIR  = os.path.join(RAW_RES_DIR, 'sim_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{connectome[-3:]}')

    if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
    if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'consensus')

    filename, class_labels, class_mapping_ctx = load_metada(connectome)
    spins = np.genfromtxt(os.path.join(DATA_DIR, 'spin_test', 'spin_' + connectome + '.csv'), delimiter=',').astype(int)

    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------

    params = []
    for iter_id in range(N_RUNS):

        tmp = {'conn_name':connectome,
               'connectome':connectome,
               'scores_file':filename,
               'iter_id':iter_id,
               'iter_conn':False,
               'iter_io':False,
               'iter_sim':False,
               'encode':True,
               'decode':False,
               'readout_modules':class_mapping_ctx.copy()[spins[:, iter_id]],
               'path_res_conn':CONN_DIR,
               'path_io':IO_TASK_DIR,
               'path_res_sim':RES_SIM_DIR,
               'path_res_tsk':RES_TSK_DIR,
               }

        params.append(tmp)

    pool = mp.Pool(processes=N_PROCESS)
    res = [pool.apply_async(run_workflow, (), p) for p in params]
    for r in res: r.get()
    pool.close()

    print ('PROCESSING TIME - SPINTEST')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")


#%% --------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():
    connectomes = [
                   'human_500',
                   'human_250'
                  ]

    for connectome in connectomes:
        reliability(connectome)
        significance(connectome)
        spintest(connectome)

if __name__ == '__main__':
    main()
