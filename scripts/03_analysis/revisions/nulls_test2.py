# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:19:01 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
from bct import clustering
from reservoir.network import nulls

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
 
#%%
CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability_mod/scale500'
NULL_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/significance_mod/scale500'

ctx = np.load('E:/P3_RC/neuromorphic_networks/data/cortical/cortical_human_500.npy')
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

labels, counts = np.unique(rsn_mapp_int, return_counts=True)

coords = np.load('E:/P3_RC/neuromorphic_networks/data/coords/coords_human_500.npy')    


#%%
import time 
print ('INITIATING PROCESSING TIME - SIGNIFICANCE UNPERTURBED')
t0_1 = time.clock()
t0_2 = time.time()

conn = np.load(os.path.join(CONN_DIR, 'consensus_0.npy')) 
new_conn, eff = nulls.randmio_but_unperturbed(conn=conn, 
                                              class_mapping=rsn_mapp, 
                                              swaps=10, 
                                              unperturbed='VIS'
                                              )

print ('PROCESSING TIME - SIGNIFICANCE UNPERTURBED')
print (time.clock()-t0_1, "seconds process time")
print (time.time()-t0_2, "seconds wall time")


#%%,
conn_vis = conn.copy()[np.ix_(np.where(rsn_mapp == 'VIS')[0], np.where(rsn_mapp == 'VIS')[0])]
new_conn_vis = new_conn.copy()[np.ix_(np.where(rsn_mapp == 'VIS')[0], np.where(rsn_mapp == 'VIS')[0])]

a = conn_vis[np.tril_indices_from(conn_vis, -1)]
b = new_conn_vis[np.tril_indices_from(new_conn_vis, -1)]

for i in range(len(a)):
    if a[i] != b[i]:
        print(i)


