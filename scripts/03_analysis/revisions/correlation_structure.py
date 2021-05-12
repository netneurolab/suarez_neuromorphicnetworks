# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:04:32 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from netneurotools import plotting


#%% --------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = 'E:/P3_RC/neuromorphic_networks' #os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')


#%% --------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_mapp_int = rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]
#alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])

n_nodes = {rsn:len(np.where(rsn_mapp == rsn)[0]) for rsn in rsn_labels}

#%% --------------------------------------------------------------------------------------------------------------------
# CORRELATION ACROSS RSNs
# ----------------------------------------------------------------------------------------------------------------------
#fc = []
#for sample_id in range(1000): 
#    reservoir_states = np.load(os.path.join(RAW_RES_DIR, 'sim_results', 'reliability', 'subctx_scale500', f'reservoir_states_{sample_id}.npy'))
#    for i, alpha in enumerate(alphas[6:7]):
#        res_states = reservoir_states[i,0,:,:]
#        fc.append(np.corrcoef(res_states.T))
#   
#fc = np.dstack(fc)
#np.save('C:/Users/User/Desktop/FC.npy', fc.squeeze())
#

#%% BOXPLOTS CORRELATIONS WITHIN RSNs
#fc = np.load('C:/Users/User/Desktop/FC.npy')
#fc = np.mean(fc, axis=2)
#
#tmp_fc = []
#for j, rsn in enumerate(rsn_labels[:7]):  
#    fc_ = fc.copy()[np.ix_(np.where(rsn_mapp == rsn)[0],np.where(rsn_mapp == rsn)[0])]
#    tmp_fc.append(fc_[np.tril_indices_from(fc_, -1)])
# 
#labels = np.row_stack([np.repeat(rsn_labels[p], len(q), axis=0)[:,np.newaxis] for p,q in enumerate(tmp_fc)])
#tmp_fc = np.row_stack([a[:,np.newaxis] for a in tmp_fc])
#
#tmp_df = pd.DataFrame(np.column_stack([labels, tmp_fc]),
#                      columns=['class', 'FC'])
#
#tmp_df['FC'] = tmp_df['FC'].astype(float)
#
#sns.set(style="ticks", font_scale=2.0) 
#fig = plt.figure(figsize=(12,8))
#ax = plt.subplot(111)
#   
#sns.boxplot(x = 'class',
#               y = 'FC', 
#               data=tmp_df,
#               palette=sns.color_palette("husl", 8),
#               hue='class',
##                       width=1.0
#               )
#
##ax.set_ylim(0.99, 1.0)
#sns.despine(offset=10, trim=True)
#plt.show()
#plt.close()
 
#%% DISTPLOTS CORRELATIONS WITHIN RSNs
fc = np.load('C:/Users/User/Desktop/FC.npy')
fc = np.mean(fc, axis=2)

COLORS = sns.color_palette("husl", 8)

sns.set(style="ticks", font_scale=2.0) 
fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for j, rsn in enumerate(rsn_labels[:7]):
    
    tmp_fc = fc.copy()[np.ix_(np.where(rsn_mapp == rsn)[0],np.where(rsn_mapp == rsn)[0])]
    sns.distplot(tmp_fc[np.tril_indices_from(tmp_fc, -1)]**2, 
                 hist=False, 
                 kde=True,
                 color=COLORS[j]
                 )
    
ax.set_xlim(0.98, 1.02)
sns.despine(offset=10, trim=True)
fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', 'FC2_dist.png'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

#%% --------------------------------------------------------------------------------------------------------------------
# RANGE OF CORRELATION WITHIN RSNs
# ----------------------------------------------------------------------------------------------------------------------
fc = np.load('C:/Users/User/Desktop/FC.npy')
#mins = []
#maxs = []
stds = []
for i in range(1000):
    
#    tmp_mins = []
#    tmp_maxs = []
    tmp_stds = []
    for j, rsn in enumerate(rsn_labels[:7]):
        tmp_fc = fc[:,:,i][np.ix_(np.where(rsn_mapp == rsn)[0],np.where(rsn_mapp == rsn)[0])]
        tmp_fc = tmp_fc[np.tril_indices_from(tmp_fc, -1)]
        
#        tmp_mins.append(tmp_fc.min())
#        tmp_maxs.append(tmp_fc.max())
    
#        tmp_mins.append((tmp_fc**2).min())
#        tmp_maxs.append((tmp_fc**2).max())
        tmp_stds.append(np.std(tmp_fc**2))


#    mins.append(tmp_mins)
#    maxs.append(tmp_maxs)
    stds.append(tmp_stds)
    
#mins = np.row_stack(mins)
#maxs = np.row_stack(maxs)
stds = np.row_stack(stds)
#ranges = maxs-mins
#np.save('C:/Users/User/Desktop/rangesFC_squared.npy', ranges.squeeze())
np.save('C:/Users/User/Desktop/stdFC_squared.npy', stds.squeeze())

#%%
ranges = np.load('C:/Users/User/Desktop/rangesFC_squared.npy')
stds = np.load('C:/Users/User/Desktop/stdFC_squared.npy')

COLORS = sns.color_palette("husl", 8)
sns.set(style="ticks", font_scale=2.0) 
fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for j, rsn in enumerate(rsn_labels[:7]):
    sns.distplot(stds[:,j], 
                 hist=False, 
                 kde=True,
                 color=COLORS[j],
                 label=rsn
                 )

plt.legend()
ax.set_xlabel('range R^2')
sns.despine(offset=10, trim=True)    
fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', 'rangeFC2_dist.png'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

#%% --------------------------------------------------------------------------------------------------------------------
# EIGENVALUES OF RSNs
# ----------------------------------------------------------------------------------------------------------------------
#mean_eig = []
#max_eig = []
#min_eig = []
#std_eig = []

mean_eigen_whole_network = []
for sample_id in range(1000):
    
    print(sample_id)
    conn = np.load(os.path.join(RAW_RES_DIR, 'conn_results', 'reliability', 'scale500', f'consensus_{sample_id}.npy'))
    conn = (conn-conn.min())/(conn.max()-conn.min())
    ew, _ = eigh(conn)
    
    conn = conn/np.max(ew)
    ew, _ = eigh(conn)

    mean_eigen_whole_network.append(np.mean(ew))

##    tmp_mean_eig = []
##    tmp_max_eig  = []
##    tmp_min_eig  = []
##    tmp_std_eig = []    
#    for rsn in rsn_labels:
#        tmp_conn = conn[np.ix_(np.where(rsn_mapp == rsn)[0],np.where(rsn_mapp == rsn)[0])]
#        ew, _ = eigh(tmp_conn)
#        
##        tmp_mean_eig.append(np.mean(ew))
##        tmp_max_eig.append(np.max(ew))
##        tmp_min_eig.append(np.min(ew))
##        tmp_std_eig.append(np.std(ew))   
#
#    
##    mean_eig.append(tmp_mean_eig)
##    max_eig.append(tmp_max_eig)
##    min_eig.append(tmp_min_eig)
##    std_eig.append(tmp_std_eig)
  
#mean_eig = np.row_stack(mean_eig)
#max_eig = np.row_stack(max_eig)
#min_eig = np.row_stack(min_eig)
#std_eig = np.row_stack(std_eig)

#np.save('C:/Users/User/Desktop/mean_eig.npy', mean_eig)
#np.save('C:/Users/User/Desktop/max_eig.npy', max_eig)
#np.save('C:/Users/User/Desktop/min_eig.npy', min_eig)
#np.save('C:/Users/User/Desktop/std_eig.npy', std_eig)
np.save('C:/Users/User/Desktop/mean_eigen_whole_network.npy', mean_eigen_whole_network)


#%%
mean_eig = np.load('C:/Users/User/Desktop/mean_eig.npy')[:,:7]
max_eig = np.load('C:/Users/User/Desktop/max_eig.npy')[:,:7]
min_eig = np.load('C:/Users/User/Desktop/min_eig.npy')[:,:7]
std_eig = np.load('C:/Users/User/Desktop/std_eig.npy')[:,:7]

range_eigen = max_eig - min_eig

#%%
COLORS = sns.color_palette("husl", 8)
sns.set(style="ticks", font_scale=2.0) 
fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

for j, rsn in enumerate(rsn_labels[:7]):
    sns.distplot(std_eig[:,j], 
                 hist=False, 
                 kde=True,
                 color=COLORS[j],
                 label=rsn
                 )

plt.legend(loc='upper right')
ax.set_xlabel('max Eigenval')
sns.despine(offset=10, trim=True)    
#fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', 'max_eigen_dist.png'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()



#%% --------------------------------------------------------------------------------------------------------------------
# ACEITUNO PLOTS
# ----------------------------------------------------------------------------------------------------------------------
#meanFC = []
#MC = []
#for sample_id in range(1000):
#    print(sample_id)
#    
#    reservoir_states = np.load(os.path.join(RAW_RES_DIR, 'sim_results', 'reliability', 'subctx_scale500', f'reservoir_states_{sample_id}.npy'))
#    df_res = pd.read_csv(os.path.join(RAW_RES_DIR, 'tsk_results', 'reliability', 'subctx_scale500', f'functional_encoding_score_{sample_id}.csv'))
#       
#    tmp_meanFC = []
#    tmp_MC = []
#    
#    for i, alpha in enumerate(alphas):
#        
#        # print(f'\n-----------{alpha}------------')
#        
#        res_states = reservoir_states[i,0,:,:]
#        fc = np.corrcoef(res_states.T)
#        tmp_meanFC.append(np.mean((fc[np.tril_indices_from(fc,-1)])**2))
#        # print(f'-----------mean FC: {np.mean((fc[np.tril_indices_from(fc,-1)])**2)}------------')
#        
#        mc = np.mean(df_res.loc[df_res['alpha'] == alpha, 'performance'].values)
#        tmp_MC.append(mc)   
#        # print(f'-----------whole brain MC: {mc}------------')
#  
#    
#    meanFC.append(tmp_meanFC)
#    MC.append(tmp_MC)
#   
#    
#np.save('C:/Users/User/Desktop/MC.npy', MC)
#np.save('C:/Users/User/Desktop/meanFC.npy', meanFC)
#
#%%
MC = np.load('C:/Users/User/Desktop/MC.npy')
meanFC = np.load('C:/Users/User/Desktop/meanFC.npy')
COLORS = sns.color_palette("viridis", 17)

sns.set(style="ticks", font_scale=2.0) 
fig = plt.figure(figsize=(8,8))

idx = 17

ax = plt.subplot(111)
x = np.mean(meanFC, axis=0)
err_x = np.std(meanFC, axis=0)
y = np.mean(MC, axis=0)
err_y = np.std(MC, axis=0)
for i, a in enumerate(alphas[:idx]):

    plt.errorbar(x=x[i],
                 y=y[i],
                 yerr=err_y[i],
                 xerr=err_x[i],
                 ecolor=COLORS[i],
                 )
    
    plt.plot(x[i], 
             y[i], 
             'D', 
             label=f'alpha = {a}', 
             c=COLORS[i], 
             markersize=15)

ax.set_ylim(4, 14)
ax.set_xlabel('<R^2>')
ax.set_ylabel('memory capacity')
sns.despine(offset=10, trim=True)

fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', 'aceitunoP1.eps'), transparent=True, bbox_inches='tight', dpi=300)

plt.show()
plt.close()

#%%
mean_eigen_whole_network = np.load('C:/Users/User/Desktop/mean_eigen_whole_network.npy')
mean_eigen_whole_network = np.array(mean_eigen_whole_network).squeeze()

sns.set(style="ticks", font_scale=2.0) 
fig = plt.figure(figsize=(17,8))

ax = plt.subplot(111)
x = alphas[:idx] # mean_eigen_whole_network[:idx] alphas[:idx] 
y = np.mean(meanFC, axis=0)
err_y = np.std(meanFC, axis=0)
for i, a in enumerate(alphas[:idx]):
    
    plt.errorbar(x=x[i],
                 y=y[i],
                 yerr=err_y[i],
                 ecolor=COLORS[i]
                 )

    plt.plot(x[i], y[i], 'D', c=COLORS[i], markersize=15)

ax.yaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.set_xlabel('<mean lambda_i>')
ax.set_ylabel('<R^2>')
sns.despine(offset=10, trim=True)

fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs/', 'aceitunoP2.eps'), transparent=True, bbox_inches='tight', dpi=300)

plt.show()
plt.close()
    
 
#%% --------------------------------------------------------------------------------------------------------------------
# FC AND DISTRIBUTION OF Rs
# ----------------------------------------------------------------------------------------------------------------------
#for sample_id in range(1):
#    
#    reservoir_states = np.load(os.path.join(RAW_RES_DIR, 'sim_results', 'reliability', 'subctx_scale500', 'reservoir_states_0.npy'))
#    df_res = pd.read_csv(os.path.join(RAW_RES_DIR, 'tsk_results', 'reliability', 'subctx_scale500', 'functional_encoding_score_0.csv'))
#       
#    for i, alpha in enumerate(alphas[:7]):
#        
#        print(f'\n-----------alpha: {alpha}------------')
#        
#        res_states = reservoir_states[i,0,:,:]
#        fc = np.corrcoef(res_states.T)
#        print(f'\tmean FC: {np.mean((fc[np.tril_indices_from(fc,-1)])**2)}------------')
#        
#        mc = np.mean(df_res.loc[df_res['alpha'] == alpha, 'performance'].values)
#        print(f'\tMC: {mc}')
#        
#        sns.set(style="ticks", font_scale=2.0) 
#        fig = plt.figure(figsize=(24,8))
#        ax = plt.subplot(121)
#    
#    #    sns.set(style="ticks", font_scale=2.0) 
#        plotting.plot_mod_heatmap(data=fc,
#                                  cmap='coolwarm',
#                                  communities=rsn_mapp_int,
#        #                          inds=None, 
#                                  edgecolor='white',
#                                  figsize=(12,8), 
#                                  xlabels=rsn_labels, 
#                                  ylabels=rsn_labels,
#                                  xlabelrotation=45, ylabelrotation=45, 
#        #                          cbar=True,
#                                  linewidth=2.5,
#                                  vmin=-1.0,
#                                  vmax=1.0,
#        #                          center=0
#                                  ax=ax
#                                 )
#        
#    
#        ax = plt.subplot(122)
#        sns.distplot(fc[np.tril_indices_from(fc,-1)])
#        sns.despine(offset=10, trim=True)
#        
#        plt.show()
#        plt.close()

#%% --------------------------------------------------------------------------------------------------------------------
# CORRELATION ACROSS RSNs
# ----------------------------------------------------------------------------------------------------------------------
#COLORS = sns.color_palette("husl", 8)
#for sample_id in range(1):
#    
#    reservoir_states = np.load(os.path.join(RAW_RES_DIR, 'sim_results', 'reliability', 'subctx_scale500', 'reservoir_states_0.npy'))
#    df_res = pd.read_csv(os.path.join(RAW_RES_DIR, 'tsk_results', 'reliability', 'subctx_scale500', 'functional_encoding_score_0.csv'))
#       
#    for i, alpha in enumerate(alphas[:8]):
#        
#        print(f'\n----------- alpha = {alpha} ------------')
#        
#        res_states = reservoir_states[i,0,:,:]
#        fc = np.corrcoef(res_states.T)
#          
#        sns.set(style="ticks", font_scale=2.0) 
#        fig = plt.figure(figsize=(12,8))
#        ax = plt.subplot(111)
#       
#        
#        for j, rsn in enumerate(rsn_labels[:7]):
#            
#            tmp_fc = fc.copy()[np.ix_(np.where(rsn_mapp == rsn)[0],np.where(rsn_mapp == rsn)[0])]
#           
#            sns.distplot(tmp_fc[np.tril_indices_from(tmp_fc, -1)], 
#                         hist=False, 
#                         kde=True,
#                         color=COLORS[j]
#                         )
#            
#        ax.set_xlim(0.98, 1.02)
#        sns.despine(offset=10, trim=True)
#        plt.show()
#        plt.close()
# 
