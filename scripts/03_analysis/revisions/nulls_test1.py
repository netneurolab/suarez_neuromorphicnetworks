# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:36:29 2021

@author: Estefany Suarez
"""

import os
import numpy as np
import bct

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
 
CONN_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/reliability_mod/scale500'
NULL_DIR = 'E:/P3_RC/neuromorphic_networks/raw_results/conn_results/significance_mod/scale500'

ctx = np.load('E:/P3_RC/neuromorphic_networks/data/cortical/cortical_human_500.npy')
rsn_mapp = np.load('E:/P3_RC/neuromorphic_networks/data/rsn_mapping/rsn_human_500.npy')
rsn_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx'])
rsn_mapp_int = np.array([np.where(rsn_labels == mapp)[0][0] for mapp in rsn_mapp])

labels, counts = np.unique(rsn_mapp_int, return_counts=True)

coords = np.load('E:/P3_RC/neuromorphic_networks/data/coords/coords_human_500.npy')

#%%   
def plot_brain(mapping, view='right'):
    sns.set(style="ticks", font_scale=1.5) 
    
    COLORS = sns.color_palette('husl', len(np.unique(mapping)))
    colors = [COLORS[mapp] for mapp in mapping]
 
    fig = plt.figure(figsize=(8, 6)) #2*.8*5,2*.8*4))#2*5.25,2*4))
    ax = plt.subplot(111, projection='3d')
    
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
               marker='o',
               s=200,
               c=colors,
               # cmap=cmap,
               # facecolors='#e6e7e8',
               # linewidths=0.8,
               edgecolors='dimgrey', #'#414042',
#               alpha=0.5,
               ) 
    
    # ax.set_title('XXXX', fontsize=10, loc=)
    ax.grid(False)
    ax.axis('off')

    if view == 'superior':
        ax.view_init(90,270)
        ax.set(xlim=0.57 *np.array(ax.get_xlim()),
               ylim=0.57 *np.array(ax.get_ylim()),
               zlim=0.60 *np.array(ax.get_zlim()),
               aspect=1.1
               )

    if view == 'left':
        ax.view_init(0,180)
        ax.set(xlim=0.59 * np.array(ax.get_xlim()),
               ylim=0.59 * np.array(ax.get_ylim()),
               zlim=0.60 * np.array(ax.get_zlim()),
               # aspect=0.55 #1.1
               )

    if view == 'right':
        ax.view_init(0,0)
        ax.set(xlim=0.59 * np.array(ax.get_xlim()),
               ylim=0.59 * np.array(ax.get_ylim()),
               zlim=0.60 * np.array(ax.get_zlim()),
               # aspect=0.55 #1.1
               )

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gca().patch.set_facecolor('white')
    sns.despine()
    
    plt.show()
    plt.close()
    


#%%
#cis_null = []
#for i, file in enumerate(os.listdir(NULL_DIR)):
#       
#    if 'rand_mio' in file:
#        conn = np.load(f'{NULL_DIR}/{file}')
#        
#        ci, q = bct.modularity.community_louvain(conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])],
#                                                 gamma=1,
#                                                 ci=rsn_mapp_int[np.where(ctx==1)[0]],
#                                                 B='modularity', 
#                                                 )
#    
#        n_comm = len(np.unique(ci))
#        print(f'\nConn #{i} - Qi = {q} with {n_comm} communities')
#        
#        new_cis = rsn_mapp_int.copy()
#        new_cis[np.where(ctx == 1)[0]] = (ci-1)
#        
#        cis_null.append(new_cis)
#        
##        plot_brain(new_cis)
#    
#labels_null, counts_null = np.unique(cis_null[0], return_counts=True)



#%%
cis_emp = []
for i, file in enumerate(os.listdir(CONN_DIR)):
         
    if 'consensus' in file:
        conn = np.load(f'{CONN_DIR}/{file}')
        
        ci, q = bct.modularity.community_louvain(conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])],
                                                 gamma=1,
                                                 ci=rsn_mapp_int[np.where(ctx==1)[0]],
                                                 B='modularity', 
                                                 )
    
        n_comm = len(np.unique(ci))
        print(f'\nConn #{i} - Qi = {q} with {n_comm} communities')
        
        new_cis = rsn_mapp_int.copy()
        new_cis[np.where(ctx == 1)[0]] = (ci-1)
              
        cis_emp.append(new_cis)
        
#        plot_brain(new_cis)

        
labels_emp, counts_emp = np.unique(cis_emp, return_counts=True)