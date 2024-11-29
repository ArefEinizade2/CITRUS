import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import scipy.sparse as sp
import tensorly as tl
from tensorly import random
from layers import CITRUS, GTCNN, GTCNN_v2
from Product_Random_Gen_ER_Noisy import Product_Random_Gen_ER, gen_Noisy_graphs
from Product_Random_Gen_ER_Noisy import get_selected_evec_evals
import os
import sys
import argparse
from statistics import mean,stdev
import torch.nn.functional as F
import scipy
import torch_geometric
from torch_geometric.utils import get_laplacian
sys.path.append(os.path.join(os.path.dirname(__file__), "diffusion_net/")) 
from load_data import get_dataset, split_data, split_data_arxive
from layers import TIDE_net
from funcs import get_optimizer, get_laplacian_selfloop, sparse_mx_to_torch_sparse_tensor
from torch_geometric.nn import GCNConv
from layers import GCN, CGNN
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import matplotlib.pyplot as plt
import time
#%%
import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import scipy.sparse as sp
import tensorly as tl
from tensorly import random
# from layers import CPGNN, GTCNN, GTCNN_v2
from Product_Random_Gen_ER_Noisy import Product_Random_Gen_ER, gen_Noisy_graphs
from Product_Random_Gen_ER_Noisy import get_selected_evec_evals
import os
import sys
import argparse
from statistics import mean,stdev
import torch.nn.functional as F
import scipy
import torch_geometric
from torch_geometric.utils import get_laplacian
sys.path.append(os.path.join(os.path.dirname(__file__), "diffusion_net/")) 
from load_data import get_dataset, split_data, split_data_arxive
from layers import TIDE_net
from funcs import get_optimizer, get_laplacian_selfloop, sparse_mx_to_torch_sparse_tensor
from torch_geometric.nn import GCNConv
from layers import GCN, CGNN
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import matplotlib.pyplot as plt
import time
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#%%
def train(epoch, optimizer, model, data, target, train_idx, mass, N, L_list, evals_list, evecs, edge_index):
                
    model.train()
    optimizer.zero_grad()
    
    # Apply the model
    out = model(epoch, data, [], mass=mass, L=L_list, evals=evals_list, evecs=evecs)
    # out = model(data, edge_index)

    # Evaluate loss
    loss = torch.nn.functional.mse_loss(out[train_idx], target[train_idx])   

    loss.backward()    # Back Propagation
    optimizer.step()   # Gardient Descent

    
    return loss

#%%
def train_GCN(epoch, optimizer, model, data, target, train_idx, mass, N, L_list, evals_list, evecs, edge_index):
                
    model.train()
    optimizer.zero_grad()
    
    # Apply the model
    out = model(data, edge_index)

    # Evaluate loss
    loss = torch.nn.functional.mse_loss(out[train_idx], target[train_idx])   

    loss.backward()    # Back Propagation
    optimizer.step()   # Gardient Descent

    
    return loss


#%%
def train_GTCNN(optimizer, model, data, target, train_idx):
                
    model.train()
    optimizer.zero_grad()
    
    # Apply the model
    out = model(data)

    # Evaluate loss
    loss = torch.nn.functional.mse_loss(out[train_idx], target[train_idx])   

    loss.backward(retain_graph=True)    # Back Propagation
    optimizer.step()   # Gardient Descent

    
    return loss
#%%
def train_GCN(epoch, optimizer, model, data, target, train_idx, mass, N, L_list, evals_list, evecs, edge_index):
                
    model.train()
    optimizer.zero_grad()
    
    # Apply the model
    out = model(data, edge_index)

    # Evaluate loss
    loss = torch.nn.functional.mse_loss(out[train_idx], target[train_idx])   

    loss.backward()    # Back Propagation
    optimizer.step()   # Gardient Descent

    
    return loss
#%%    

@torch.no_grad()
def test(epoch, model, data, target, train_idx, val_idx, test_idx, mass, N, L_list, evals_list, evecs, edge_index):
    model.eval()            
    losses = []
    with torch.no_grad():    
        out = model(epoch, data, [], mass=mass, L=L_list, evals=evals_list, evecs=evecs)
        # out = model(data, edge_index)
    for mask in [train_idx, val_idx, test_idx]:
        loss = torch.nn.functional.mse_loss(out[mask], target[mask]) 
        losses.append(loss)
    return losses
#%%
@torch.no_grad()
def test_GTCNN(model, data, target, train_idx, val_idx, test_idx):
    model.eval()            
    losses = []
    with torch.no_grad():    
        out = model(data)
    for mask in [train_idx, val_idx, test_idx]:
        loss = torch.nn.functional.mse_loss(out[mask], target[mask]) 
        losses.append(loss)
    return losses
#%%
@torch.no_grad()
def test_GCN(epoch, model, data, target, train_idx, val_idx, test_idx, mass, N, L_list, evals_list, evecs, edge_index):
    model.eval()            
    losses = []
    with torch.no_grad():    
        out = model(data, edge_index)
    for mask in [train_idx, val_idx, test_idx]:
        loss = torch.nn.functional.mse_loss(out[mask], target[mask]) 
        losses.append(loss)
    return losses
#%% Generate data:
t_total = time.time()

p_ER = [0.1, 0.1]
N = [20, 30]
# F_in = 5
# k = [F_in, F_in, F_in]
k = np.array(N)-2
SNR = torch.inf
SNR = np.inf
test_size = 0.15
val_size = 0.15
Num_layers = 3
t_exp = [2, 3]
t = [t_exp, t_exp, t_exp]
Fea = [6, 5, 4, 2]
iterations = 10
N_block = 2
hidden_f = 2
epochs = 500
lr = 1e-2
weight_decay = False
C_width = 4
k_list = [N[0] - 2, N[1] - 2]

SNR_G1 = [np.inf, 20, 10, 0, -10]
# SNR_G1 = [np.inf, -10]
SNR_G2 = SNR_G1

train_ls = np.zeros((len(SNR_G1), len(SNR_G2), iterations))
val_ls = np.zeros((len(SNR_G1), len(SNR_G2), iterations))
test_ls = np.zeros((len(SNR_G1), len(SNR_G2), iterations))

#%%
for i in range(iterations):
    X, X_noisy, Y, Y_noisy, train_idx, val_idx, test_idx, evals, evecs, L_list, Adj_Cart, Adj_list, L_Cart = Product_Random_Gen_ER(t, p_ER, N, Fea, k, SNR, test_size, val_size).gen_data(Num_layers)
    X = X.to(device)
    X_noisy = X_noisy.to(device)
    Y = Y.to(device)
    Y_noisy = Y_noisy.to(device)
    mass = torch.ones(np.prod(N)).to(device)
    #%%    
    for j in range(len(SNR_G1)):
        for kk in range(len(SNR_G2)):
            SNR_graph_list = [SNR_G1[j], SNR_G2[kk]]
            # SNR_graph_list = [-20, -20]
            L_list_noisy, Adj_list_noisy, Adj_Cart_noisy, L_Cart_noisy  = gen_Noisy_graphs(Adj_list, SNR_graph_list)
            evals, evecs = get_selected_evec_evals(L_list_noisy, k_list)
            
            
            evals_prod, evecs_prod = sparse.linalg.eigs(L_Cart_noisy.numpy(), k=np.prod(N)-2, return_eigenvectors=True)
            evals_prod = torch.tensor(evals_prod.real)
            evals_prod = evals_prod.to(torch.float32).to(device)
            evecs_prod = torch.tensor(evecs_prod.real).to(torch.float32).to(device)        
            
            
            edge_index, edge_weight = from_scipy_sparse_matrix(sparse.coo_matrix(L_Cart_noisy))
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            ## Run mode
            
            Fea_GCN = [Fea[0]]        
            for _ in range(N_block):
                Fea_GCN.append(C_width)
            Fea_GCN.append(Fea[-1])
                
                
            #%############################################ CPGNN ####################################################
            optimizer = 'adamax'    
                    
            model = CITRUS(k=np.prod(k), C_in=Fea[0], C_out=Fea[-1], C_width=C_width, num_nodes = N,
                      N_block = N_block, single_t=False, use_gdc=[],
                        last_activation=lambda x : x,
                        diffusion_method='spectral',
                        with_MLP = True,
                        dropout=True,
                        device = device)
                        
            model = model.to(device)
            
            model_CPGNN = model
                
            parameters = [p for p in model.parameters() if p.requires_grad]
        
            parameters_name= [ name for (name, param) in model.named_parameters() if param.requires_grad]
        
            # Move to device
            mass=torch.ones(np.prod(N)).to(device)
            for ii in range(len(evals)):
                evals[ii] = evals[ii].to(device)
            evecs=evecs.to(device)
        
        
            optimizer = get_optimizer(optimizer, parameters, lr = lr, weight_decay=weight_decay)      
        
        
            total_train=[]
            total_test=[]
            total_val=[]
        
            best_loss = train_loss = val_loss = test_loss = torch.inf
        
            t_CPGNN = time.time()

            for epoch in range(1, epochs + 1):
               
                loss = train(epoch, optimizer, model, X_noisy, Y, train_idx, mass, N=N, L_list=L_list, 
                             evals_list=evals, evecs=evecs, edge_index=edge_index)
                
                tmp_train_loss, tmp_val_loss, tmp_test_loss = test(epoch, model, X_noisy, Y, train_idx, val_idx, test_idx, mass, 
                                                                   N=N, L_list=L_list, 
                                                                   evals_list=evals, evecs=evecs, edge_index=edge_index)
                
                if tmp_val_loss < val_loss:
                    best_epoch = epoch
                    train_loss = tmp_train_loss
                    val_loss = tmp_val_loss
                    test_loss = tmp_test_loss
                      
                
                    total_train.append(train_loss.cpu().numpy())
                    total_val.append(val_loss.cpu().numpy())
                    total_test.append(test_loss.cpu().numpy())
                    
                    # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
                    # print(f'CPGNN >>>> iter {i}, SNR_G1: {SNR_G1[j]}, SNR_G2: {SNR_G2[kk]}, Step {epoch}: ' f' Loss: {float(loss):.4f}, Train Loss: {train_loss:.4f}, 'f'Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
                    
        
            elapsed = time.time() - t_CPGNN
            print('>>>>>>>>>>>>>>>>>>>> CPGNN training time, Elapsed: %s' % round(elapsed/60,2), ' minutes')

            train_ls[j, kk, i] = train_loss.cpu().numpy()
            val_ls[j, kk, i] = val_loss.cpu().numpy()
            test_ls[j, kk, i] = test_loss.cpu().numpy()
            print(f'CPGNN >>>> iter {i}, SNR_G1: {SNR_G1[j]}, SNR_G2: {SNR_G2[kk]}, Step {epoch}: ' f' Test Loss: {test_loss:.4f}')
            
            
            plt.figure()
            plt.close()
            plt.plot(total_train,'-')
            plt.plot(total_val,'-')
            plt.plot(total_test,'-')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['Train','Valid','Test'])
            plt.title('l: '+str(N_block) + ', MSE: ' + str(np.round(total_test[-1], 2)) + ', SNRs: ' + str(SNR_graph_list))
            plt.show()
    
    
    #%
    
print()

print(f'CPGNN >>>> Average:')
print(np.round(np.mean(test_ls, -1), 2))  
print()


print(f'CPGNN >>>> Standard deviation:')
print(np.round(np.std(test_ls, -1), 2))  
print()

#%% Save and Load the stability results:
np.save('Stability_Results_NEW.npy', test_ls)
a = np.load('Stability_Results_NEW.npy')
#%%
# total_params = sum(p.numel() for p in model_CPGNN.parameters())
# print(f">>>>>>>>>>>>>> Number of parameters: {total_params} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# if iterations>1:
#     print(f'Test Loss: {np.mean(test_ls, -1):2.2f}\xB1{np.std(test_ls):.1e}')
#     print(f'Test Loss: {np.mean(test_ls_CGNN, -1):2.2f}\xB1{np.std(test_ls_CGNN):.1e}')

elapsed = time.time() - t_total
print('>>>>>>>>>>>>>>>>>>>> TOTAL time, Elapsed: %s' % round(elapsed/60,2), ' minutes')
















