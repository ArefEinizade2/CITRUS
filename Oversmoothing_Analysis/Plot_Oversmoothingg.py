import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm
import networkx as nx
from numpy import linalg as LA
from scipy import sparse
import torch
#%%
n_layers = 10
n_layers_list = [0] + list(np.arange(n_layers+1))
n_layers_list = n_layers_list[1:]


#%%
low_p = 0.05; high_p = 0.95
p_ER = [low_p, high_p]

a = np.load('Oversmoothing_Results_1.npz')
E_list = a['array1']
Therorem_bound = a['array2']
Lambda = a['array3']
s_max_list = a['array4']
#%%

fig, axs = plt.subplots(1, 2, layout='constrained')
axs[0].plot(n_layers_list, E_list, label='Actual')
# axs[i, j].plot(n_layers_list, E_list_GCN, label='Actual, GCN')
axs[0].plot(n_layers_list, Therorem_bound, label='Theorem', ls='--')
axs[0].set_xlabel('# layer')
axs[0].set_ylabel('Log relative dist')
# axs[0].set_ylim([-100, 0])
axs[0].grid(True)
axs[0].legend(handlelength=4)
axs[0].set_title('p: ' + str(p_ER) + ', l: ' + str(np.round(Lambda,2)) 
                 + ', s: ' 
                 + str(np.round(np.mean(s_max_list), 3)) )
                 # + ', std = ' + str(np.round(np.std(s_max_list), 5)))
axs[0].set_box_aspect(1)
#%%
low_p = 0.1; high_p = 0.95
p_ER = [low_p, low_p]

a = np.load('Oversmoothing_Results_2.npz')
E_list = a['array1']
Therorem_bound = a['array2']
Lambda = a['array3']
s_max_list = a['array4']

#%%

axs[1].plot(n_layers_list, E_list, label='Actual')
# axs[i, j].plot(n_layers_list, E_list_GCN, label='Actual, GCN')
axs[1].plot(n_layers_list, Therorem_bound, label='Theorem', ls='--')
axs[1].set_xlabel('# layer')
axs[1].set_ylabel('Log relative dist')
# axs[1].set_ylim([-100, 0])
axs[1].grid(True)
axs[1].legend(handlelength=4)
axs[1].set_title('p: ' + str(p_ER) + ', l: ' + str(np.round(Lambda,2)) 
                 + ', s: ' 
                 + str(np.round(np.mean(s_max_list), 3)) )
                 # + ', std = ' + str(np.round(np.std(s_max_list), 5)))
axs[1].set_box_aspect(1)

plt.savefig('OverSmoothing.png')
plt.savefig("OverSmoothing.pdf")
plt.savefig("OverSmoothing.eps")
plt.savefig("OverSmoothing.svg")
plt.show()


