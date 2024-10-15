import networkx as nx
import numpy as np
import random
import torch
from scipy import sparse
from diffusion_net.geometry import to_basis, from_basis
from sklearn.model_selection import train_test_split
#%%
def get_evcs_evals(Graph_list, K_list):
    
    P = len(Graph_list)
    
    # G = self.gen_connected_ER(self.N[0], self.p_ER[0])
    L = nx.laplacian_matrix(Graph_list[0]).toarray()
    
    # Compute degree matrix
    A = nx.to_numpy_array(Graph_list[0])  
    A = A + 1e-10*np.eye(A.shape[0])              
    degrees = np.sum(A, axis=1)
    # D = np.diag(degrees)
    
    # Compute normalized Laplacian
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
    L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
    L_normalized = L_normalized/P
    L_sparse = sparse.coo_matrix(L_normalized)
    L_normalized_sparse_list = [L_sparse]
    
    print(K_list)
    
    evals, evecs = sparse.linalg.eigs(L_sparse, k=K_list[0], return_eigenvectors=True)
    evals = torch.tensor(evals.real)
    evals = evals.to(torch.float32)
    evals_list = [evals]
    evecs=torch.tensor(evecs.real)        
    evecs_kron = evecs
    print('evecs.shape:, ', evecs.shape)
    
    for p in range(1, P):
        
        L = nx.laplacian_matrix(Graph_list[p]).toarray()

        # Compute degree matrix
        A = nx.to_numpy_array(Graph_list[p])
        degrees = np.sum(A, axis=1)
        # D = np.diag(degrees)

        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/P
        L_sparse = sparse.coo_matrix(L_normalized)
        L_normalized_sparse_list.append(L_sparse)

        evals, evecs = sparse.linalg.eigs(L_sparse, k=K_list[p], return_eigenvectors=True)
        evals = torch.tensor(evals.real)
        evals = evals.to(torch.float32)
        evals_list.append(evals)
        evecs = torch.tensor(evecs.real)        
        print('evecs.shape:, ', evecs.shape)
        evecs_kron = torch.kron(evecs_kron, evecs)
        print('evecs_kron.shape:, ', evecs.shape)

    return evecs_kron.to(torch.float32), evals_list, L_normalized_sparse_list
#%%
def get_selected_evec_evals(L_normalized_sparse_list, k_list):
    evals, evecs = sparse.linalg.eigs(L_normalized_sparse_list[0], k=k_list[0], return_eigenvectors=True)
    evals = torch.tensor(evals.real)
    evals = evals.to(torch.float32)
    evals_list = [evals]
    evecs=torch.tensor(evecs.real).to(torch.float32)        
    evecs_kron = evecs
    
    for p in range(1, len(L_normalized_sparse_list)):

        evals, evecs = sparse.linalg.eigs(L_normalized_sparse_list[p], k=k_list[p], return_eigenvectors=True)
        evals = torch.tensor(evals.real)
        evals = evals.to(torch.float32)
        evals_list.append(evals)
        evecs = torch.tensor(evecs.real)        
        evecs_kron = torch.kron(evecs_kron, evecs).to(torch.float32)
    
    return evals_list, evecs_kron
#%%
class Product_Random_Gen_ER:
    # income_graph_data should be list of graph initialization data, for more information check nx.Graph() documentation
    def __init__(self, t, p_ER, N, Fea, k, SNR, test_size, val_size):
        self.t = t
        self.N = N
        self.P = len(self.N)
        self.p_ER = p_ER
        # self.F_in = F_in
        # self.F_out = F_out
        self.Fea = Fea
        # self.F.append(F_out)
        self.k = k
        self.SNR = SNR
        self.test_size = test_size
        self.val_size = val_size
#%%   
    def gen_X_W(self, Num_layers):
        X = torch.randn(np.prod(np.array(self.N)), self.Fea[0])
        W = [torch.randn(self.Fea[0], self.Fea[1]).to(torch.float32)]
        for i in range(1, Num_layers):
            W.append(torch.randn(self.Fea[i], self.Fea[i+1]).to(torch.float32))
        return X.to(torch.float32), W
#%%
    def Cartesian_Product(self, A, B):
        C = torch.kron(A, torch.eye(B.shape[0])) + torch.kron(torch.eye(A.shape[0]), B)
        return C
    
    def gen_connected_ER(self, n, p):
        connected = False
        while not connected:
            G = nx.erdos_renyi_graph(n, p)
            connected = nx.is_connected(G)
        return G
    #%%
    
    def gen_factor_graphs(self):
        
        # G = self.gen_connected_ER(self.N[0], self.p_ER[0])
        G = nx.path_graph(self.N[0])
        L = nx.laplacian_matrix(G).toarray()
        
        # Compute degree matrix
        A = nx.to_numpy_array(G)        
        degrees = np.sum(A, axis=1)
        # D = np.diag(degrees)
        
        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/self.P
        L_sparse = sparse.coo_matrix(L_normalized)
        L_normalized_sparse_list = [L_sparse]
        
        evals, evecs = sparse.linalg.eigs(L_sparse, k=self.k[0], return_eigenvectors=True, tol=5e-1)
        evals = torch.tensor(evals.real)
        evals = evals.to(torch.float32)
        evals_list = [evals]
        evecs=torch.tensor(evecs.real)        
        evecs_kron = evecs
        
        for p in range(1, self.P):
            # G = self.gen_connected_ER(self.N[p], self.p_ER[p])
            num_nodes_per_block = [9, 9, 8]  # Number of nodes in each block
            num_blocks = len(num_nodes_per_block)
            p_in = 0.8  # Probability of edge within a block
            p_out = 0.05  # Probability of edge between blocks
            
            # Generate SBM graph
            G = nx.generators.community.stochastic_block_model(num_nodes_per_block, [[p_in, p_out, p_out],
                                                                                   [p_out, p_in, p_out],
                                                                                   [p_out, p_out, p_in]])

            L = nx.laplacian_matrix(G).toarray()

            # Compute degree matrix
            Adj_Cart = self.Cartesian_Product(torch.tensor(A), torch.tensor(nx.to_numpy_array(G)))
            A = nx.to_numpy_array(G)
            degrees = np.sum(A, axis=1)
            # D = np.diag(degrees)

            # Compute normalized Laplacian
            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
            L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
            L_normalized = L_normalized/self.P
            L_sparse = sparse.coo_matrix(L_normalized)
            L_normalized_sparse_list.append(L_sparse)

            evals, evecs = sparse.linalg.eigs(L_sparse, k=self.k[p], return_eigenvectors=True)
            evals = torch.tensor(evals.real)
            evals = evals.to(torch.float32)
            evals_list.append(evals)
            evecs = torch.tensor(evecs.real)        
            evecs_kron = torch.kron(evecs_kron, evecs)
            
        return evecs_kron.to(torch.float32), evals_list, L_normalized_sparse_list, Adj_Cart
    #%%
    def gen_data(self, Num_layers):
        evecs_kron, evals, L_normalized_sparse_list, Adj_Cart = self.gen_factor_graphs()
        X, W = self.gen_X_W(Num_layers)
        # Transform to spectral        
        
        x_diffuse = X
        # Diffuse
        for i in range(Num_layers):
            x_spec = to_basis(x_diffuse, evecs_kron, torch.ones(np.prod(np.array(self.N))))
            t_temp = self.t[i]    
            evals_kroned = t_temp[0] * evals[0]
            for p  in range(1, self.P):
                evals_kroned = torch.kron(evals_kroned, t_temp[p] * evals[p])
            evals_kroned = evals_kroned.repeat(self.Fea[i], 1).T
            diffusion_coefs = torch.exp(-evals_kroned)
            x_diffuse_spec = diffusion_coefs * x_spec
    
            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec.to(torch.float32), evecs_kron)        
            Y = torch.tanh(torch.matmul(x_diffuse, W[i]))
            x_diffuse = Y
            
        Noise = torch.randn(X.shape)
        alpha = torch.pow(torch.tensor(10), -self.SNR/20)*torch.norm(x_diffuse, 'fro')/torch.norm(Noise, 'fro')
        X_noisy = X + alpha * Noise


        Noise = torch.randn(Y.shape)
        alpha = torch.pow(torch.tensor(10), -self.SNR/20)*torch.norm(Y, 'fro')/torch.norm(Noise, 'fro')
        Y_noisy = Y + alpha * Noise

        train_idx, test_idx, train_idx2, test_idx2 = train_test_split(range(x_diffuse.shape[0]),
                                                            range(x_diffuse.shape[0]),
                                                            test_size=self.test_size)
        
        train_idx, val_idx, train_idx2, val_idx2 = train_test_split(range(x_diffuse.shape[0]),
                                                            range(x_diffuse.shape[0]),
                                                            test_size=self.val_size)

        return X, X_noisy, Y, Y_noisy, train_idx, val_idx, test_idx, evals, evecs_kron, L_normalized_sparse_list, Adj_Cart
        
        
        
        
        
        
        
        
        
