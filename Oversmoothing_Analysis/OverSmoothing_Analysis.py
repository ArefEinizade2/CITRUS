import numpy as np
from scipy.linalg import expm, sinm, cosm
import networkx as nx
from numpy import linalg as LA
from scipy import sparse
import torch
import matplotlib.pyplot as plt
#%%
def Cartesian_Product(A, B):
    C = np.kron(A, np.eye(B.shape[0])) + np.kron(np.eye(A.shape[0]), B)
    return C
#%%
def gen_connected_ER(n, p):
    connected = False
    while not connected:
        G = nx.erdos_renyi_graph(n, p)
        connected = nx.is_connected(G)
    return G
#%%
def gen_factor_graphs(N, p_ER, k):
    
    Adj_list = []
    P = len(N)
    
    G = gen_connected_ER(N[0], p_ER[0])
    L = nx.laplacian_matrix(G).toarray()
    # Compute degree matrix
    A = nx.to_numpy_array(G)  
    Adj_Cart = A
    Adj_list.append(A)
    degrees = np.sum(A, axis=1)
    # D = np.diag(degrees)
    
    # Compute normalized Laplacian
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
    Adj_normalized = np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
    L_normalized = np.eye(N[0]) - Adj_normalized
    # L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
    L_normalized = L_normalized/P
    L_Cart = L_normalized
    L_normalized_list = [L_normalized]
    # L_sparse = sparse.coo_matrix(L_normalized)
    evals, evecs = LA.eig(L_normalized)
    evals = evals.real
    # evals = evals.to(torch.float32)
    evals_list = [evals]
    evecs=evecs.real        
    evecs_kron = evecs
    
    for p in range(1, P):
        G = gen_connected_ER(N[p], p_ER[p])

        # Compute degree matrix
        Adj_Cart = Cartesian_Product(Adj_Cart, nx.to_numpy_array(G))
        L = nx.laplacian_matrix(G).toarray()
        A = nx.to_numpy_array(G)
        Adj_list.append(A)
        degrees = np.sum(A, axis=1)
        # D = np.diag(degrees)

        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        Adj_normalized = np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        L_normalized = np.eye(N[p]) - Adj_normalized
        # L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/P
        L_Cart = Cartesian_Product(L_Cart, L_normalized)
        L_normalized_list.append(L_normalized)
        # L_sparse = sparse.coo_matrix(L_normalized)
        evals, evecs = LA.eig(L_normalized)
        evals = evals.real
        # evals = evals.to(torch.float32)
        evals_list.append(evals)
        evecs = evecs.real        
        evecs_kron = np.kron(evecs_kron, evecs)
    evals_Cart, evecs_Cart = LA.eig(L_Cart)
    evals_Cart = evals_Cart.real
    evecs_Cart = evecs_Cart.real        
    
    return evecs_kron, evals_list, L_normalized_list, Adj_Cart, Adj_list, L_Cart, evals_Cart, evecs_Cart
#%%
def relu(x):
    return np.maximum(0, x)
#%%
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)
#%% matrix exponentials:
a = expm(np.zeros((2, 2)))

#%% maximum singular value:

a = np.arange(12)
a
b = a.reshape((3, 4))
U, S, Vh = np.linalg.svd(b, full_matrices=True)
LA.norm(b, ord=2)
#%% genereate graphs:
def Oversmoothing_analysis(N, p_ER, k, t, n_layers_list, Norm_coeff, Fea, X0):
    # N = [50, 60]
    # p_ER = [0.9, 0.9]
    # k = [2, 2]
    evecs_kron, evals_list, L_normalized_sparse_list, Adj_Cart, Adj_list, L_Cart, evals_Cart, evecs_Cart = gen_factor_graphs(N, p_ER, k)
    # Adj_Cart = np.eye(np.prod(N)) - L_Cart
    # a = np.sum(Adj_Cart, 0)
    
    #%%
    evals_Cart_sorted = np.sort(evals_Cart)
    evals_Cart_ll = evals_Cart_sorted[evals_Cart_sorted>1e-10]
    Lambda = np.min(evals_Cart_ll)
    # L_Cart2 = np.eye(np.prod(N)) - Adj_Cart # Just to check
    n_layers = 20; n_MLP = 5; 
    
    
    n_layers_list = n_layers_list[1:]
    
    n_layers = len(n_layers_list)
    
    
    # Fea = np.arange(start=n_layers+2, stop=1, step=-1)
    # # Fea = (np.mean(n_layers_list)*np.ones(n_layers+1)).astype('int32') 
    
    # X0 = np.random.randn(np.prod(N), Fea[0])
    
    E_X0 = np.trace(X0.T@L_Cart@X0)
    
    E_list = [0]
    
    E_list_GCN = [0]
    
    Therorem_bound = [0]
    
    s_max_list = []
    
    X_l = X0
    
    X_l_GCN = X0
    
    s_list = []
    
    for l in range(n_layers):
        # print(l)
        X_l = expm(-t*L_Cart)@X_l
        X_l_GCN = Adj_Cart@X_l_GCN
        s_MLP = 1
        
        for ll in range(1):
            W_l = np.random.randn(Fea[l], Fea[l+1])/Norm_coeff
            s_MLP = s_MLP * (LA.norm(W_l, ord=2)**2)
            X_l = relu(X_l@W_l)
            X_l_GCN = relu(X_l_GCN@W_l)
            
        s_list.append(s_MLP)
        
        E_list.append(np.log(np.trace(X_l.T@L_Cart@X_l)/E_X0))
        
        E_list_GCN.append(np.log(np.trace(X_l_GCN.T@L_Cart@X_l_GCN)/E_X0))
    
        s_max = np.max(np.array(s_list))
        
        s_max_list.append(s_max)
        
        Therorem_bound.append((l+1)*(np.log(s_max)-t*(Lambda)))
        
    return E_list, E_list_GCN, Therorem_bound, Lambda, s_max_list
#%%
low_p = 0.3; high_p = 0.9
p_ER_list = [[low_p, low_p], [low_p, high_p], [high_p, low_p], [high_p, high_p]]
N = [10, 15]
k = [2, 2]
t = 7
Norm_coeff = 100

# n_layers_list = [0, 1, 2, 4, 8, 16, 32, 64]
# n_layers = len(n_layers_list)
n_layers = 10
n_layers_list = [0] + list(np.arange(n_layers+1))
fig, axs = plt.subplots(2, 2, layout='constrained')
cc = 0
n_layers_list = n_layers_list[1:]

Fea = np.arange(start=n_layers+2, stop=1, step=-1)
# Fea = (np.mean(n_layers_list)*np.ones(n_layers+1)).astype('int32') 

X0 = np.random.randn(np.prod(N), Fea[0])

for p_ER in p_ER_list:

    E_list, E_list_GCN, Therorem_bound, Lambda, s_max_list = Oversmoothing_analysis(N, p_ER, 
                                                                                    k, t, 
                                                                                    n_layers_list, 
                                                                                    Norm_coeff,
                                                                                    Fea, X0)
    i = int(cc/2)
    j = cc - i*2
    axs[i, j].plot(n_layers_list, E_list, label='Actual')
    # axs[i, j].plot(n_layers_list, E_list_GCN, label='Actual, GCN')
    axs[i, j].plot(n_layers_list, Therorem_bound, label='Theorem', ls='--')
    axs[i, j].set_xlabel('# layer')
    axs[i, j].set_ylabel('relative dist')
    axs[i, j].set_ylim([-100, 0])
    axs[i, j].grid(True)
    axs[i, j].legend(handlelength=4)
    axs[i, j].set_title('p: ' + str(p_ER) + ', lamb: ' + str(np.round(Lambda,2)) 
                     + ', s: ' 
                     + str(np.round(np.mean(s_max_list), 3)) )
                     # + ', std = ' + str(np.round(np.std(s_max_list), 5)))
    
    cc+=1
    
plt.show()
#%%
low_p = 0.05; high_p = 0.95
p_ER = [low_p, high_p]
N = [10, 15]
k = [2, 2]
t = 1

Norm_coeff = 100

# n_layers_list = [0, 1, 2, 4, 8, 16, 32, 64]
# n_layers = len(n_layers_list)
n_layers = 10
n_layers_list = [0] + list(np.arange(n_layers+1))
n_layers_list = n_layers_list[1:]

Fea = np.arange(start=n_layers+2, stop=1, step=-1)
# Fea = (np.mean(n_layers_list)*np.ones(n_layers+1)).astype('int32') 

X0 = np.random.randn(np.prod(N), Fea[0])

E_list, E_list_GCN, Therorem_bound, Lambda, s_max_list = Oversmoothing_analysis(N, p_ER, 
                                                                                k, t, 
                                                                                n_layers_list, 
                                                                                Norm_coeff,
                                                                                Fea, X0)

np.savez('Oversmoothing_Results_1.npz', array1=E_list, array2=Therorem_bound, array3=Lambda, array4=s_max_list)
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
Norm_coeff = 2.5
E_list, E_list_GCN, Therorem_bound, Lambda, s_max_list = Oversmoothing_analysis(N, p_ER, 
                                                                                k, t, 
                                                                                n_layers_list, 
                                                                                Norm_coeff,
                                                                                Fea, X0)
#%%
np.savez('Oversmoothing_Results_2.npz', array1=E_list, array2=Therorem_bound, array3=Lambda, array4=s_max_list)
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
