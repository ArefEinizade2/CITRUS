import networkx as nx
import numpy as np
import random
import torch
from scipy import sparse
from diffusion_net.geometry import to_basis, from_basis
from sklearn.model_selection import train_test_split

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
        
        Adj_list = []
        
        G = self.gen_connected_ER(self.N[0], self.p_ER[0])
        L = nx.laplacian_matrix(G).toarray()
        L_Cart = L
        # Compute degree matrix
        A = nx.to_numpy_array(G)  
        Adj_Cart = A
        Adj_list.append(A)
        degrees = np.sum(A, axis=1)
        # D = np.diag(degrees)
        
        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/self.P
        L_sparse = sparse.coo_matrix(L_normalized)
        L_normalized_sparse_list = [L_sparse]
        
        evals, evecs = sparse.linalg.eigs(L_sparse, k=self.k[0], return_eigenvectors=True)
        evals = torch.tensor(evals.real)
        evals = evals.to(torch.float32)
        evals_list = [evals]
        evecs=torch.tensor(evecs.real)        
        evecs_kron = evecs
        
        for p in range(1, self.P):
            G = self.gen_connected_ER(self.N[p], self.p_ER[p])

            # Compute degree matrix
            Adj_Cart = self.Cartesian_Product(torch.tensor(Adj_Cart), torch.tensor(nx.to_numpy_array(G)))
            L_Cart = self.Cartesian_Product(torch.tensor(L_Cart), torch.tensor(nx.laplacian_matrix(G).toarray()))
            L = nx.laplacian_matrix(G).toarray()
            A = nx.to_numpy_array(G)
            Adj_list.append(A)
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
            
        return evecs_kron.to(torch.float32), evals_list, L_normalized_sparse_list, Adj_Cart, Adj_list, L_Cart
    #%%
    def gen_data(self, Num_layers):
        evecs_kron, evals, L_normalized_sparse_list, Adj_Cart, Adj_list, L_Cart = self.gen_factor_graphs()
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
        
        train_idx, val_idx, train_idx2, val_idx2 = train_test_split(train_idx,
                                                            train_idx2,
                                                            test_size=self.val_size)

        return X, X_noisy, Y, Y_noisy, train_idx, val_idx, test_idx, evals, evecs_kron, L_normalized_sparse_list, Adj_Cart, Adj_list, L_Cart   
        
#%%
def Cartesian_Product(A, B):
    A = torch.tensor(A)
    B = torch.tensor(B)
    C = torch.kron(A, torch.eye(B.shape[0])) + torch.kron(torch.eye(A.shape[0]), B)
    return C

#%%
def gen_Noisy_graphs(Adj_list, SNR_list):
    
    P = len(Adj_list)  
    # Compute degree matrix
    A = Adj_list[0]
    E = np.random.rand(A.shape[0], A.shape[0])
    E = (E+E.T)/2
    np.fill_diagonal(E, 0)
    alpha = np.power(10, -SNR_list[0]/20)*np.linalg.norm(A, 'fro')/np.linalg.norm(E, 'fro')
    A_noisy = A + alpha*E
    # degrees = np.sum(A_noisy, axis=1)
    # inverse_sqrt_degree_matrix = np.linalg.inv(np.sqrt(np.diag(degrees)))
    # A_noisy = np.dot(np.dot(inverse_sqrt_degree_matrix, A_noisy), inverse_sqrt_degree_matrix)/P
    Adj_Cart = A_noisy
    Adj_list_Noisy = [A_noisy]
    degrees = np.sum(A_noisy, axis=1)
    L = np.diag(degrees) - A_noisy
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
    L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
    L_normalized = L_normalized/P
    L_sparse = sparse.coo_matrix(L_normalized)
    L_normalized_sparse_list = [L_normalized]
    L_Cart = L_normalized
    
    
    for p in range(1, P):

        A = Adj_list[p]
        E = np.random.rand(A.shape[0], A.shape[0])
        E = (E+E.T)/2
        np.fill_diagonal(E, 0)
        alpha = np.power(10, -SNR_list[p]/20)*np.linalg.norm(A, 'fro')/np.linalg.norm(E, 'fro')
        A_noisy = A + alpha*E
        # degrees = np.sum(A_noisy, axis=1)
        # inverse_sqrt_degree_matrix = np.linalg.inv(np.sqrt(np.diag(degrees)))
        # A_noisy = np.dot(np.dot(inverse_sqrt_degree_matrix, A_noisy), inverse_sqrt_degree_matrix)/P
        Adj_Cart = Cartesian_Product(torch.tensor(Adj_Cart), A_noisy)
        Adj_list_Noisy.append(A_noisy)
        degrees = np.sum(A_noisy, axis=1)
        L = np.diag(degrees) - A_noisy
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/P
        L_sparse = sparse.coo_matrix(L_normalized)
        L_normalized_sparse_list.append(L_normalized)
        L_Cart = Cartesian_Product(torch.tensor(L_Cart), torch.tensor(L_normalized))
    
    return L_normalized_sparse_list, Adj_list_Noisy, Adj_Cart, L_Cart       
        
        
        
        
        
        
        
