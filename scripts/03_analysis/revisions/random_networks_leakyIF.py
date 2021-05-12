# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:52:00 2021

@author: Estefany Suarez
"""
from reservoir.network import nulls
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#%%
coords = np.load('E:/P3_RC/neuromorphic_networks/data/coords/coords_human_500.npy')
euc_dist = cdist(coords, coords, metric='euclidean')


#%%
lambda_ = 2.0
#p = np.exp(-(euc_dist/lambda_)**2)
#np.fill_diagonal(p,0) 
#p = 1/((euc_dist/lambda_)**2)
#p = 1/((euc_dist/lambda_))

p = 1/euc_dist

#fig = plt.figure(figsize=(10,10))
#ax = plt.subplot(111)
#
#plt.imshow(np.log(p))
#plt.show()
#plt.close()



#%%
np.fill_diagonal(p,0) 
p = (p-np.min(p[np.nonzero(p)]))/(np.max(p[np.nonzero(p)])-np.min(p[np.nonzero(p)]))
np.fill_diagonal(p,0) 
p[p<=0] = 0
print(np.mean(p))


#%%
plt.hist(p[np.nonzero(p)], bins=100)
#plt.xlim(0.5,1.0)
plt.show()


#%%
W = np.random.binomial(1, p, p.shape).astype(bool).astype(int)#*np.random.uniform(-1,1,p.shape)
upper_diag = W.copy()[np.triu_indices_from(W,1)]
W = W.T
W[np.triu_indices_from(W,1)] = upper_diag

print(nulls.check_symmetric(W))
print(np.sum(W.astype(bool).astype(int))/(len(W)**2))





#%%
#G = nx.fast_gnp_random_graph(1015, 0.025, seed=None, directed=False)
#W = nx.to_numpy_array(G).astype(int)
#
##%%
#W = W*np.random.uniform(-1,1,W.shape)
#
#
##%%
#W[np.where(abs(W) <= 0.00001)] = 0
#
##%%
#upper_diag = W.copy()[np.triu_indices_from(W,1)]
#W = W.T
#W[np.triu_indices_from(W,1)] = upper_diag
#np.fill_diagonal(W,0) 
#
#print(nulls.check_symmetric(W))
#print(np.sum(W.astype(bool).astype(int))/(len(W)**2))
#

