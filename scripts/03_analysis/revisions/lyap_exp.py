# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:32:44 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
iter_id = 0

#DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/tsk_results'
wb = np.load(f'E:/P3_RC/neuromorphic_networks/raw_results/tsk_results/lyap_exp_whole_brain/subctx_scale500/distance_{iter_id}.npy')
thr = np.load(f'E:/P3_RC/neuromorphic_networks/raw_results/tsk_results/lyap_exp_threshold/subctx_scale500/distance_{iter_id}.npy')


#%%
sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(figsize=(24,8))
alphas_wb = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05,\
             1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

t_ini = 0
t_end = 100
t_pert = 1

ax = plt.subplot(131)
for i, dist in enumerate(wb[:6]): 
    lambda_ = np.round(np.log(np.max(dist)/dist[t_pert]), 4)
    z = plt.plot(np.arange(t_ini, t_end),
                 dist[t_ini:t_end], 
                 label = f'alpha:{alphas_wb[i]}  lambda:{lambda_}'
                 )
    plt.legend()
    
ax = plt.subplot(132)
for i, dist in enumerate(wb[6:11]): 
    lambda_ = np.round(np.log(np.max(dist)/dist[t_pert]), 4)
    z = plt.plot(np.arange(t_ini, t_end),
                 dist[t_ini:t_end], 
                 label = f'alpha:{alphas_wb[i+6]}  lambda:{lambda_}'
                 )
    plt.legend()
    
ax = plt.subplot(133)
for i, dist in enumerate(wb[11:]): 
    lambda_ = np.round(np.log(np.max(dist)/dist[t_pert]), 2)
    z = plt.plot(np.arange(t_ini, t_end),
                 dist[t_ini:t_end], 
                 label = f'alpha:{alphas_wb[i+11]}  lambda:{lambda_}'
                 )
    plt.legend()
    
    
plt.show()
plt.close()


#%%
sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(figsize=(24,8))
alphas_thr=[.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,\
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10]

t_ini = 0
t_end = 100
t_pert = 0

ax = plt.subplot(131)
for i, dist in enumerate(thr[:9]): 
    lambda_ = np.round(np.log(dist[t_pert+1]/dist[t_pert]),2)
    x = plt.plot(np.arange(t_ini, t_end),
                 dist[t_ini:t_end], 
                 label = f'alpha:{alphas_thr[i]}  lambda:{lambda_}'
                 )
    plt.legend()
    
ax = plt.subplot(132)
for i, dist in enumerate(thr[9:11]): 
    lambda_ = np.round(np.log(dist[t_pert+1]/dist[t_pert]), 2)
    x = plt.plot(np.arange(t_ini, t_end),
                 dist[t_ini:t_end], 
                 label = f'alpha:{alphas_thr[i+9]}  lambda:{lambda_}'
                 )
    plt.legend()
    
ax = plt.subplot(133)
for i, dist in enumerate(thr[11:]): 
    lambda_ = np.round(np.log(dist[t_pert+1]/dist[t_pert]), 2)
    x = plt.plot(np.arange(t_ini, t_end),
                 dist[:t_end], 
                 label = f'alpha:{alphas_thr[i+11]}  lambda:{lambda_}'
                 )
    plt.legend()
    
    
plt.show()
plt.close()

