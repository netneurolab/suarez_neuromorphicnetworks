
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
whole_brain = pd.read_csv('E:/P3_RC/neuromorphic_networks/proc_results/tsk_results/memory_capacity/whole_brain/subctx_scale500/functional_avg_encoding.csv')
leakyIF = pd.read_csv('E:/P3_RC/neuromorphic_networks/proc_results/tsk_results/memory_capacity/leakyIF/subctx_scale500/functional_avg_encoding.csv')
lsm = np.unique(np.loadtxt('C:/Users/User/Dropbox/lsm.txt'))
lsm = np.hstack((np.arange(len(lsm))[:,np.newaxis], np.repeat(['LSM'], len(lsm), axis=0)[:,np.newaxis], lsm[:,np.newaxis], np.repeat(['LSM'], len(lsm), axis=0)[:,np.newaxis]))

#%%
df_wb  = whole_brain[['sample_id', 'alpha', 'performance', 'analysis']]
df_lk  = leakyIF[['sample_id', 'alpha', 'performance', 'analysis']]
df_lsm = pd.DataFrame(data=lsm, columns=['sample_id', 'alpha', 'performance', 'analysis'])

new_df = pd.concat([df_wb, df_lk, df_lsm], axis=0).reset_index()[['sample_id', 'alpha', 'performance', 'analysis']]
new_df['performance'] = new_df['performance'].astype(float)

#%%
min_score = np.min(new_df['performance'])
max_score = np.max(new_df['performance'])
new_df['performance'] = (new_df['performance']-min_score)/(max_score-min_score)


#%%
#new_df['alpha'] = new_df['alpha'].astype(str)
#include_alpha = ['0.9', 'LSM']
#new_df = pd.concat([new_df.loc[new_df['alpha'] == alpha, :] for alpha in include_alpha])\
#                    .reset_index(drop=True)
#
#new_df['performance'] = (new_df['performance']-np.min(new_df['performance']))/(np.max(new_df['performance'])-np.min(new_df['performance']))
            
#%%
plotting.boxplot(x='alpha', y='performance', df=new_df.copy(),
#                 palette=sns.color_palette('husl', ),
#                 suptitle=score,
                 hue='analysis',
                 order=None,
                 xlim=None,
                 ylim=(-0.1,1.1),
#                 legend=True,
                 width=0.7,
#                 fig_name=f'brain_vs_nulls_vs_alpha_{CONNECTOME}_{INPUTS}',
                 figsize=(22,8),
                 showfliers=True,
                 fig_name='other_vs_brain'
                 )
