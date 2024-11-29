import torch
import numpy as np
from scipy import sparse
from Utils.layers import CITRUS
import networkx as nx

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
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
def gen_connected_ER(n, p):
    connected = False
    while not connected:
        G = nx.erdos_renyi_graph(n, p)
        connected = nx.is_connected(G)
    return G
#%%
def gen_factor_graphs(N_list, p_list):
    
    P = len(N_list)
    
    Adj_list = []
    
    G = gen_connected_ER(N_list[0], p_list[0])
    L = nx.laplacian_matrix(G).toarray()
    
    # Compute degree matrix
    A = nx.to_numpy_array(G)  
    Adj_list.append(A)
    degrees = np.sum(A, axis=1)
    # D = np.diag(degrees)
    
    # Compute normalized Laplacian
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
    L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
    L_normalized = L_normalized/P
    L_sparse = sparse.coo_matrix(L_normalized)
    L_normalized_sparse_list = [L_sparse]
    
    
    for p in range(1, P):
        G = gen_connected_ER(N_list[p], p_list[p])
        L = nx.laplacian_matrix(G).toarray()

        # Compute degree matrix
        A = nx.to_numpy_array(G)
        Adj_list.append(A)
        degrees = np.sum(A, axis=1)
        # D = np.diag(degrees)

        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        L_normalized = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        L_normalized = L_normalized/P
        L_sparse = sparse.coo_matrix(L_normalized)
        L_normalized_sparse_list.append(L_sparse)

        
    return L_normalized_sparse_list
#%%
def mode_n_matricization(tensor, mode):
    """
    Perform mode-n matricization of a tensor.

    Parameters:
    tensor (ndarray): The input tensor of shape (I1, I2, ..., IN).
    mode (int): The mode along which to matricize (0-based index).

    Returns:
    ndarray: The mode-n matricized tensor.
    """
    # Move the mode-th axis to the first dimension
    tensor = np.moveaxis(tensor, mode, 0)
    # Get new shape: (size of mode-n, product of remaining dimensions)
    new_shape = (tensor.shape[0], -1)
    # Reshape tensor to new shape
    matricized_tensor = tensor.reshape(new_shape)
    return matricized_tensor
#%%
def reverse_mode_n_matricization(matricized_tensor, original_shape, mode):
    """
    Reverse mode-n matricization to reconstruct the original tensor.

    Parameters:
    matricized_tensor (ndarray): The matricized tensor of shape (I_n, prod(I_1, ..., I_{n-1}, I_{n+1}, ..., I_N)).
    original_shape (tuple): The original shape of the tensor (I1, I2, ..., IN).
    mode (int): The mode along which the matricization was performed (0-based index).

    Returns:
    ndarray: The reconstructed tensor with the original shape.
    """
    # Determine the shape after expanding back to the original tensor's dimensions
    new_shape = (original_shape[mode],) + tuple(dim for i, dim in enumerate(original_shape) if i != mode)
    
    # Reshape the matricized tensor back to this expanded shape
    reshaped_tensor = matricized_tensor.reshape(new_shape)
    
    # Reverse the axis reordering to get back to the original shape
    reconstructed_tensor = np.moveaxis(reshaped_tensor, 0, mode)
    return reconstructed_tensor
#%% Generate the graphs:
p_list = [0.3, 0.3, 0.05]
N_list = [10, 15, 20]
L_list = gen_factor_graphs(N_list, p_list)
# F_in = 5
# k = [F_in, F_in, F_in]
k = np.array(N_list)-2
SNR = torch.inf
SNR = 0
test_size = 0.85
val_size = 0.85
Num_layers = 3
t = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
Fea = [4, 2]
N_block = 32
iteration = 2
epochs = 100
lr = 1e-2
weight_decay = False
C_width = 4
k_list = [N_list[0] - 2, N_list[1] - 2, N_list[2] - 2]
train_ls = []
val_ls = []
test_ls = []
N_block_list = [1, 2, 4, 8, 16, 32]
#%%
X_tensor = torch.randn(N_list[0], N_list[1], N_list[2], Fea[0])

X = torch.tensor(mode_n_matricization(X_tensor.numpy(), 3).T).to(device)

evals, evecs = get_selected_evec_evals(L_list, k_list)
## Run mode
            
optimizer = 'adamax'

model = CITRUS(k=np.prod(k), C_in=Fea[0], C_out=Fea[-1], C_width=C_width, num_nodes = N_list,
          N_block = 4, single_t=False, use_gdc=[],
            last_activation=lambda x : x,
            diffusion_method='spectral',
            with_MLP = True,
            dropout=True,
            device = device)
            
model = model.to(device)


model_CITRUS = model
    
parameters = [p for p in model.parameters() if p.requires_grad]

parameters_name= [ name for (name, param) in model.named_parameters() if param.requires_grad]

# Move to device
mass=torch.ones(np.prod(N_list)).to(device)

for ii in range(len(evals)):
    evals[ii] = evals[ii].to(device)
evecs=evecs.to(device)

out = model(0, X, [], mass=mass, L=L_list, evals=evals, evecs=evecs) #GITHUUUUUUUUUB


out_tensor = reverse_mode_n_matricization(out.cpu().detach().numpy(), (N_list[0], N_list[1], N_list[2], Fea[-1]), 3)


