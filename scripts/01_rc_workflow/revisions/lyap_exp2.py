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
import bct
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

INPUTS = 'subctx'
CLASS = 'functional' #'functional' 'cytoarch'

N_PROCESS = 1 #40
N_RUNS = 1 #000

CONNECTOME = 'human_500'

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def calculate_distance(a, b, nodes):
    dist = []
    for state_a, state_b in zip(a,b):
        state_a = state_a.squeeze()[:, nodes]
        state_b = state_b.squeeze()[:, nodes]

        # distance in time
        tmp_dist = [np.linalg.norm(sa-sb) for sa,sb in zip(state_a, state_b)]

        dist.append(tmp_dist)
    return dist


def whole_brain(iter_id, input_len=None):

    print ('INITIATING PROCESSING TIME - LYAPUNOV EXP - WHOLE BRAIN')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP_CONN = 'reliability'
    EXP = 'lyap_exp_whole_brain'

    CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP_CONN, f'scale{CONNECTOME[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{CONNECTOME[-3:]}')

    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # import connectivity data
    conn = np.load(os.path.join(CONN_DIR,  f'consensus_{iter_id}.npy'))

    # normalize connectivity matrix by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # define set of input and output nodes
    ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + CONNECTOME + '.npy'))
    input_nodes = np.where(ctx == 0)[0]
    output_nodes = np.where(ctx == 1)[0]

    # create input signal
    if input_len is None: input_len = 100
    inputs = np.zeros((input_len, n_nodes))
    inputs[:,input_nodes] = np.repeat(np.random.uniform(-1, 1, (input_len))[:,np.newaxis], len(input_nodes), axis=1)

    # simulate network IC = 0
    init_states = sim_lnm.run_sim(conn=conn,
                                  input_nodes=input_nodes,
                                  inputs=inputs,
                                  factor=0.0001,
                                  task=SPEC_TASK,
                                  activation='tanh',
                                  )

    # simulate network with perturbed ICs
    pert_states = sim_lnm.run_sim(conn=conn,
                                  input_nodes=input_nodes,
                                  inputs=inputs,
                                  factor=0.0001,
                                  task=SPEC_TASK,
                                  activation='tanh',
                                  add_perturb=True,
                                  t_perturb=200
                                 )

    #  estimate distance between states across alpha
    dist = calculate_distance(init_states, pert_states, output_nodes)

    # save distance vector
    np.save(os.path.join(RES_TSK_DIR, f'distance_{iter_id}.npy'), dist)

    print ('PROCESSING TIME - LYAPUNOV EXP - WHOLE BRAIN')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")

def threshold(iter_id, input_len=None):

    print ('INITIATING PROCESSING TIME - LYAPUNOV EXP - THRESHOLD')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP_CONN = 'reliability_thr'
    EXP = 'lyap_exp_threshold'

    CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP_CONN, f'scale{CONNECTOME[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{CONNECTOME[-3:]}')

    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # import connectivity data
    conn = np.load(os.path.join(CONN_DIR,  f'consensus_{iter_id}.npy'))

    # normalize connectivity matrix by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # define set of input and output nodes
    ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + CONNECTOME + '.npy'))
    input_nodes = np.where(ctx == 0)[0]
    output_nodes = np.where(ctx == 1)[0]

    # create input signal
    if input_len is None: input_len = 100
    inputs = np.zeros((input_len, n_nodes))
    inputs[:,input_nodes] = np.repeat(np.random.uniform(-1, 1, (input_len))[:,np.newaxis], len(input_nodes), axis=1)

    # simulate network IC = 0
    init_states = sim_lnm.run_sim(conn=conn,
                                  input_nodes=input_nodes,
                                  inputs=inputs,
                                  factor=0.1,
                                  task=SPEC_TASK,
                                  activation='piecewise',
                                  threshold=1.0,
                                  alphas=[.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,\
                                         5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10]
                                  )

    # simulate network with perturbed ICs
    pert_states = sim_lnm.run_sim(conn=conn,
                                  input_nodes=input_nodes,
                                  inputs=inputs,
                                  factor=0.1,
                                  task=SPEC_TASK,
                                  activation='piecewise',
                                  threshold=1.0,
                                  alphas=[.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,\
                                         5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10],
                                  add_perturb=True,
                                  t_perturb=200
                                 )

    #  estimate distance between states across alpha
    dist = calculate_distance(init_states, pert_states, output_nodes)

    # save distance vector
    np.save(os.path.join(RES_TSK_DIR, f'distance_{iter_id}.npy'), dist)

    print ('PROCESSING TIME - LYAPUNOV EXP - THRESHOLD')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")

def leakyIF(iter_id, input_len=None):

    print ('INITIATING PROCESSING TIME - LYAPUNOV EXP - LEAKY IF')
    # t0_1 = time.clock()
    # t0_2 = time.time()

    EXP_CONN = 'leakyIF'
    EXP = 'lyap_exp_leakyIF'

    CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP_CONN, f'scale{CONNECTOME[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP, f'{INPUTS}_scale{CONNECTOME[-3:]}')

    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # import connectivity data
    conn = np.load(os.path.join(CONN_DIR,  f'erdos_renyi_{iter_id}.npy'))

    # normalize connectivity matrix by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # define set of input and output nodes
    if input_nodes is None: input_nodes = np.random.choice(n_nodes, 15, replace=False)
    if output_nodes is None: output_nodes = np.setdiff1d(np.arange(n_nodes), input_nodes)

    # create input signal
    input_len = 100
    inputs = np.zeros((input_len, n_nodes))
    inputs[:,input_nodes] = np.repeat(np.random.uniform(-1, 1, (input_len))[:,np.newaxis], len(input_nodes), axis=1)

    # simulate network IC = 0
    ics = np.zeros((n_nodes))
    init_states = sim_lnm.run_sim_oger(conn=conn,
                                       input_nodes=input_nodes,
                                       inputs=input_train,
                                       factor=FACTOR,
                                       task=SPEC_TASK,
                                       my_initial_state=ics
                                       )

    # simulate network with perturbed ICs
    ics[np.random.choice(n_nodes, 1)] = np.random.uniform(-1, 1, (1))[0]
    pert_states = sim_lnm.run_sim_oger(conn=conn,
                                       input_nodes=input_nodes,
                                       inputs=input_train,
                                       factor=FACTOR,
                                       task=SPEC_TASK,
                                       my_initial_state=ics
                                       )

    #  estimate distance between states across alpha
    dist = calculate_distance(init_states, pert_states, output_nodes)

    # save distance vector
    np.save(os.path.join(RES_TSK_DIR, f'distance_{iter_id}.npy'), dist)

    print ('PROCESSING TIME - LYAPUNOV EXP - WHOLE BRAIN')
    # print (time.clock()-t0_1, "seconds process time")
    # print (time.time()-t0_2, "seconds wall time")


#%% --------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():

    # whole brain
    params = []
    for iter_id in range(1):
        tmp_params_dict = {'iter_id':iter_id,
                           'input_len':500,
                           }

        params.append(tmp_params_dict)

    pool1 = mp.Pool(processes=N_PROCESS)
    res1 = [pool1.apply_async(whole_brain, (), p) for p in params[:]]
    for r1 in res1: r1.get()
    pool1.close()

    # threshold
    params = []
    for iter_id in range(1):
        tmp_params_dict = {'iter_id':iter_id,
                           'input_len':500,
                           }

        params.append(tmp_params_dict)

    pool2 = mp.Pool(processes=N_PROCESS)
    res2 = [pool2.apply_async(threshold, (), p) for p in params[:]]
    for r2 in res2: r2.get()
    pool2.close()

if __name__ == '__main__':
    main()
