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
N = [20, 35]
# F_in = 5
# k = [F_in, F_in, F_in]
k = np.array(N)-2
SNR = torch.inf
SNR = np.inf
test_size = 0.85
val_size = 0.85
Num_layers = 3
t_exp = [2, 3]
t = [t_exp, t_exp, t_exp]
iterations = 10
N_block = 2
hidden_f = 2
epochs = 500
lr = 1e-2
weight_decay = False
C_width = 4
k_list = [N[0] - 2, N[1] - 2]

SNR_G1 = [np.inf, 20]
# SNR_G1 = [np.inf, -10]
SNR_G2 = SNR_G1
SNR_graph_list = [10, 10]

tt_total = np.zeros((4, 2, iterations))
train_ls = np.zeros((4, iterations))
val_ls = np.zeros((4, iterations))
test_ls = np.zeros((4, iterations))


last_f_list = [2, 5, 8, 11]
#%%
for i in range(iterations):
    for idx, last_f in enumerate(last_f_list):
        Fea = [6, 5, 4, last_f]
        X, X_noisy, Y, Y_noisy, train_idx, val_idx, test_idx, evals, evecs, L_list, Adj_Cart, Adj_list, L_Cart = Product_Random_Gen_ER(t, p_ER, N, Fea, k, SNR, test_size, val_size).gen_data(Num_layers)
        X = X.to(device)
        X_noisy = X_noisy.to(device)
        Y = Y.to(device)
        Y_noisy = Y_noisy.to(device)
        mass = torch.ones(np.prod(N)).to(device)
        #%%    
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
    
        parameters = [p for p in model.parameters() if p.requires_grad]
    
        parameters_name= [ name for (name, param) in model.named_parameters() if param.requires_grad]

        t1 = parameters[6]
        t1 = t1.cpu().detach().numpy()
        t2 = parameters[15]
        t2 = t2.cpu().detach().numpy()
        tt = np.mean(np.abs(np.concatenate((t1, t2), -1)),-1)
        tt_total[idx, :, i] = tt

        train_ls[idx, i] = train_loss.cpu().numpy()
        val_ls[idx, i] = val_loss.cpu().numpy()
        test_ls[idx, i] = test_loss.cpu().numpy()
        print(f'CPGNN >>>> iter {i}, Last_f: {last_f}, Step {epoch}: ' f' Test Loss: {test_loss:.4f}')
        
        
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
# np.save('Stability_Results.npy', test_ls)
# a = np.load('Stability_Results.npy')
#%%
# total_params = sum(p.numel() for p in model_CPGNN.parameters())
# print(f">>>>>>>>>>>>>> Number of parameters: {total_params} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# if iterations>1:
#     print(f'Test Loss: {np.mean(test_ls, -1):2.2f}\xB1{np.std(test_ls):.1e}')
#     print(f'Test Loss: {np.mean(test_ls_CGNN, -1):2.2f}\xB1{np.std(test_ls_CGNN):.1e}')

import numpy as np
import matplotlib.pyplot as plt

SNR_list = SNR_G1
p_list = range(2)

# a = np.load('Stability_Results.npy')

#%%%
plt.figure()
plt.plot(last_f_list, np.mean(test_ls, 1))
# for idx, last_f in enumerate(last_f_list):
#     Err_Mat_CGPGNN_temp = np.squeeze(np.flip(test_ls[idx, :], 0))
#     Mean_error_CGPGNN = np.mean(Err_Mat_CGPGNN_temp)
#     Mean_error_CGPGNN_var = np.var(Err_Mat_CGPGNN_temp, axis=1)*1.5
#     plt.plot(p_list, Mean_error_CGPGNN, label='last_f='+str(last_f))
#     plt.fill_between(p_list, Mean_error_CGPGNN - Mean_error_CGPGNN_var, 
#                      Mean_error_CGPGNN + Mean_error_CGPGNN_var, alpha=0.2)
#     plt.xlabel('SNR_1')
#     plt.ylabel('prediction error')    
# # plt.legend(['SNR_2=inf', 'SNR_2=20', 'SNR_2=10', 'SNR_2=0', 'SNR_2=-10'])  
# plt.legend()  
# plt.grid(True)
# # plt.xticks(range(5), ['inf', '20', '10', '0', '-10'])
# plt.xticks(range(len(last_f_list)), ['2', '5', '8', '11'])
    
# plt.savefig('Stability_Results.png')
# fig.savefig('Stability_Results.pdf')
# fig.savefig('Stability_Results.eps')
# fig.savefig('Stability_Results.svg')
plt.show() 
#%%
fig = plt.figure()
for idx in range(2):
    Err_Mat_CGPGNN_temp = np.squeeze(np.flip(tt_total[:, idx, :], 0))
    Mean_error_CGPGNN = np.mean(Err_Mat_CGPGNN_temp, axis=1)
    Mean_error_CGPGNN_var = np.var(Err_Mat_CGPGNN_temp, axis=1)*1.5
    plt.plot(last_f_list, Mean_error_CGPGNN, label='factor graph #'+str(idx))
    plt.fill_between(last_f_list, Mean_error_CGPGNN - Mean_error_CGPGNN_var, 
                     Mean_error_CGPGNN + Mean_error_CGPGNN_var, alpha=0.2)
    plt.xlabel('horizon')
    plt.ylabel('learned t')    
# plt.legend(['SNR_2=inf', 'SNR_2=20', 'SNR_2=10', 'SNR_2=0', 'SNR_2=-10'])  
plt.title('b = ' + str(7.08))
plt.legend()  
plt.grid(True)
# plt.xticks(range(5), ['inf', '20', '10', '0', '-10'])
# plt.xticks(range(4), ['2', '5', '8', '11'])
    
plt.savefig('Learned_t.png')
fig.savefig('Learned_t.pdf')
fig.savefig('Learned_t.eps')
fig.savefig('Learned_t.svg')
plt.show() 
#%%
elapsed = time.time() - t_total
print('>>>>>>>>>>>>>>>>>>>> TOTAL time, Elapsed: %s' % round(elapsed/60,2), ' minutes')
#%%
t1 = parameters[6]
t1 = t1.cpu().detach().numpy()
t2 = parameters[15]
t2 = t2.cpu().detach().numpy()








