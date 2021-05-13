
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:36:29 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
from reservoir.tasks import tasks

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from plotting import plotting

#%% --------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
CONNECTOME = 'human_500'
CLASS = 'functional' #'functional' 'cytoarch'
INPUTS = 'subctx'
ANALYSIS = 'reliability' # 'reliability' 'significance' 'spintest'


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')
RES_TSK_DIR = os.path.join(PROC_RES_DIR, 'tsk_results')

brain   = pd.read_csv(os.path.join(RES_TSK_DIR, 'reliability', 'subctx_scale500', 'functional_avg_encoding.csv'))
leakyIF = pd.read_csv(os.path.join(RES_TSK_DIR, 'leakyIF', 'subctx_scale500', 'functional_avg_encoding.csv'))
lsm = np.unique(np.loadtxt(os.path.join(RES_TSK_DIR, 'lsm', 'lsm.txt')))
lsm = np.hstack((np.arange(len(lsm))[:,np.newaxis], np.repeat(['lsm'], len(lsm), axis=0)[:,np.newaxis], lsm[:,np.newaxis], np.repeat(['lsm'], len(lsm), axis=0)[:,np.newaxis]))


#%% --------------------------------------------------------------------------------------------------------------------
# IMPORT RESULTS
# ----------------------------------------------------------------------------------------------------------------------
df_br  = brain[['sample_id', 'alpha', 'performance', 'analysis']]
df_br['analysis'] = 'brain'

df_lk  = leakyIF[['sample_id', 'alpha', 'performance', 'analysis']]
df_lsm = pd.DataFrame(data=lsm, columns=['sample_id', 'alpha', 'performance', 'analysis'])

new_df = pd.concat([df_br, df_lk, df_lsm], axis=0).reset_index()[['sample_id', 'alpha', 'performance', 'analysis']]
new_df['performance'] = new_df['performance'].astype(float)

# scale values between 0 and 1 
min_score = np.min(new_df['performance'])
max_score = np.max(new_df['performance'])
new_df['performance'] = (new_df['performance']-min_score)/(max_score-min_score)

# select alpha = 1.0  
new_df['alpha'] = new_df['alpha'].astype(str)
include_alpha = ['1.0', 'lsm']
new_df = pd.concat([new_df.loc[new_df['alpha'] == alpha, :] for alpha in include_alpha])\
                    .reset_index(drop=True)


#%% --------------------------------------------------------------------------------------------------------------------
# BRAIN vs LEAKY INTEGRATE-FIRE vs LIQUID STATE MACHINE
# ----------------------------------------------------------------------------------------------------------------------
plotting.boxplot(x='analysis', y='performance', df=new_df.copy(),
                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(-0.1,1.1),
                 width=0.7,
                 figsize=(8,8),
                 showfliers=True,
                 )
