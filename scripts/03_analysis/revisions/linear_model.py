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

#%%
tsk_res = pd.read_csv('E:/P3_RC/neuromorphic_networks/proc_results/tsk_results/memory_capacity/reliability/subctx_scale500/functional_avg_encoding.csv')

#%%
mc_lm = []
for _ in range(1000):
    
    signal = np.random.uniform(-1, 1, (2050))
    signal = np.vstack((signal,signal))
    
    r, _ = tasks.run_mem_cap(signal, signal)
    
    mc_lm.append(np.sum(r))

alpha_col = np.repeat(['linear model'], 1000, axis=0)
mc_lm = np.hstack((np.arange(1000)[:,np.newaxis], alpha_col[:,np.newaxis],np.array(mc_lm)[:,np.newaxis]))

#%%
df_lm = pd.DataFrame(data=mc_lm, columns=['sample_id', 'alpha', 'performance'])
df_rs = tsk_res[['sample_id', 'alpha', 'performance']]

new_df = pd.concat([df_rs, df_lm], axis=0).reset_index()[['sample_id', 'alpha', 'performance']]
new_df['performance'] = new_df['performance'].astype(float)
new_df['performance'] = (new_df['performance']-np.min(new_df['performance']))/(np.max(new_df['performance'])-np.min(new_df['performance']))


#%%
plotting.boxplot(x='alpha', y='performance', df=new_df.copy(),
#                 palette=sns.color_palette('husl', ),
#                 suptitle=score,
#                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(-0.1,1.1),
#                 legend=True,
                 width=0.5,
#                 fig_name=f'brain_vs_nulls_vs_alpha_{CONNECTOME}_{INPUTS}',
                 figsize=(25,10),
                 showfliers=True,
                 fig_name='linearmodel_vs_reservoir'
                 )
