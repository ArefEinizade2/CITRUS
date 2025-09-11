#%%
import time
t = time.time()
#%%
M = 6
M_hat = M
n_epochs = 300
val_len = 0.1
test_len = 0.1
lr = 1e-2
batch_size = 2048
enable_progress_bar = True
horizon = 3
#%%
# Install required packages.
from torchmetrics.regression import MeanAbsolutePercentageError
import os
import torch
from tsl.metrics.torch import MaskedMSE, MaskedMAE, MaskedMAPE
from tsl.engines import Predictor

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

import shutil
# Get the current script file path
script_path = os.path.realpath(__file__)
# Specify the destination file path
destination_path = './MetrLA_Results/'

if not os.path.isdir(destination_path):  
    os.mkdir(destination_path)
# Copy the current script to the destination
shutil.copy2(script_path, destination_path)

#%%
from layers import CITRUS
import networkx as nx
from Utilsss import get_evcs_evals
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tsl.data import SpatioTemporalDataset
import torch.nn as nn
import tsl
import torch
import torch_geometric
import numpy as np
import pandas as pd

print(f"torch version: {torch.__version__}")
print(f"  PyG version: {torch_geometric.__version__}")
print(f"  tsl version: {tsl.__version__}")
#%%
# Plotting functions ###############
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)
#%%
# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)
#%%
def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

#%%
from tsl.datasets import MetrLA

dataset = MetrLA(root='./MetrLA')

print(dataset)

#%%
print(f"Sampling period: {dataset.freq}")
print(f"Has missing values: {dataset.has_mask}")
print(f"Has exogenous variables: {dataset.has_covariates}")
print(f"Covariates: {', '.join(dataset.covariates.keys())}")

print_matrix(dataset.dist)
dataset.dataframe()
#%%
print(f"Default similarity: {dataset.similarity_score}")
print(f"Available similarity options: {dataset.similarity_options}")
print("==========================================")

sim = dataset.get_similarity("distance")  # or dataset.compute_similarity()

print("Similarity matrix W:")
print_matrix(sim)
#%%
connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        layout="edge_index",
                                        force_symmetric=True)

edge_index, edge_weight = connectivity

print(f'edge_index {edge_index.shape}:\n', edge_index)
print(f'edge_weight {edge_weight.shape}:\n', edge_weight)
#%%
from tsl.ops.connectivity import edge_index_to_adj

adj = edge_index_to_adj(edge_index, edge_weight)
print(f'A {adj.shape}:')
print_matrix(adj)
print(f'Sparse edge weights:\n', adj[edge_index[1], edge_index[0]])
#%%
torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                      connectivity=connectivity,
                                      mask=dataset.mask,
                                      horizon=horizon,
                                      window=M_hat,
                                      stride=1)
print(torch_dataset)
#%%
sample = torch_dataset[0]
# torch_dataset2 = torch_dataset[:1000]
print(sample)
#%%
a = sample.input.to_dict()

b = sample.target.to_dict()

if sample.has_mask:
    print(sample.mask)
else:
    print("Sample has no mask.")

if sample.has_transform:
    print(sample.transform)
else:
    print("Sample has no transformation functions.")
#%%
print(sample.pattern)
print("==================   Or we can print patterns and shapes together   ==================")
print(sample)
#%%
batch = torch_dataset[:5]
print(batch)
#%%
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data.preprocessing import StandardScaler

# Normalize data using mean and std computed over time and node dimensions
scalers = {'target': StandardScaler(axis=(0, 1))}

# Split data sequentially:
#   |------------ dataset -----------|
#   |--- train ---|- val -|-- test --|
splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=batch_size,
)

# print(dm)
#%%
dm.setup()
print(dm)

#%%
emb_size = 16    #@param
hidden_size = 32   #@param
rnn_layers = 1     #@param
gnn_kernel = 2   #@param

input_size = torch_dataset.n_channels   # 1 channel
n_nodes = torch_dataset.n_nodes         # 207 nodes
horizon = torch_dataset.horizon         # 12 time steps

#%% CGP-GNN:
N = [n_nodes, M]
K_list = list(np.array(N)-2)
K_list = [205, M-2]

Graph_List = [nx.from_numpy_array(np.array(adj)), nx.path_graph(N[1])]

evecs, evals, L_list = get_evcs_evals(Graph_List, K_list)

for ii in range(len(evals)):
    evals[ii] = evals[ii].to(device)
    
    
CGP_GNN = CITRUS(input_size=input_size,
                            n_nodes=n_nodes,
                            horizon=horizon,
                            emb_size=emb_size,
                            hidden_size=hidden_size,
                            rnn_layers=rnn_layers,
                            gnn_kernel=gnn_kernel,
                            edge_index=torch.tensor(edge_index).to(device),
                            edge_weight=torch.tensor(edge_weight).to(device),
                            mass = torch.ones(np.prod(N)).to(device),
                            evals = evals,
                            evecs = torch.tensor(evecs).to(device),
                            C_width = 64,
                            N_block = 3,
                            single_t = True,
                            use_gdc = [],
                            num_nodes = N,
                            last_activation=torch.nn.ReLU(), 
                            mlp_hidden_dims=[64, 64, 64, 64], 
                            dropout=False, 
                            with_MLP=True, 
                            diffusion_method='spectral', 
                            device = device,
                            graph_wise=False)
              
print(CGP_GNN)
print_model_size(CGP_GNN)

loss_fn = MaskedMAE()

metrics = {'mse': MaskedMSE(),
           'mae': MaskedMAE(),
           'mape': MaskedMAPE()}


# setup predictor_CGP_GNN
# setup predictor
predictor_CGP_GNN = Predictor(
    model=CGP_GNN,                   # our initialized model
    optim_class=torch.optim.Adam,  # specify optimizer to be used...
    optim_kwargs={'lr': lr},    # ...and parameters for its initialization
    loss_fn=loss_fn,               # which loss function to be used
    metrics=metrics,
# metrics to be logged during train/val/test
)
logger_CGP_GNN = WandbLogger(project="CITRUS_continuous_graph_product", name="MetrLA", version="0")

checkpoint_callback_CGPGNN = ModelCheckpoint(
    dirpath='FINAL_MetrLA_M6_H3',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

trainer_CGP_GNN = pl.Trainer(max_epochs=n_epochs,
                      logger=logger_CGP_GNN,
                      accelerator='gpu',
                      devices=[1], 
#                      limit_train_batches=train_batches,  # end an epoch after 100 updates
                      callbacks=[checkpoint_callback_CGPGNN],
                      enable_progress_bar=enable_progress_bar)

t_CGPGNN = time.time()
trainer_CGP_GNN.fit(predictor_CGP_GNN, datamodule=dm)
elapsed = time.time() - t_CGPGNN
print('>>>>>>>>>>>>>>>>>>>> CGP-GNN training time, Elapsed: %s' % round(elapsed/60,2), ' minutes')

predictor_CGP_GNN.load_model(checkpoint_callback_CGPGNN.best_model_path)
predictor_CGP_GNN.freeze()



CGP_GNN_results = trainer_CGP_GNN.test(predictor_CGP_GNN, datamodule=dm);


#% Detailed metrics:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XX = dm.testset
x_test = XX[:].x
y_test = XX[:].y
edge_index_test = XX[:].edge_index
edge_weight_test = XX[:].edge_weight

#CGPGNN:
print(10*'*')
print('CGPGNN:')    
y_pred = trainer_CGP_GNN.predict(predictor_CGP_GNN, dm.test_dataloader())
a = [y_pred[i]['y_hat'] for i in range(len(y_pred))]
y_pred = torch.cat(a, axis=0)
loss = nn.L1Loss()
MAE = loss(y_pred, y_test)
print(MAE)
MAPE = MeanAbsolutePercentageError()
MAPE = MAPE(y_pred, y_test)
print(MAPE)
loss = nn.MSELoss()
MSE = loss(y_pred, y_test)
print(MSE)
RelativeMAE = MAE/torch.abs(y_test).mean()
print(RelativeMAE)
RelativeMSE = MSE/((y_test**2).mean())
print(RelativeMSE)
Metrics_CGPGNN = [MAE.numpy(), MAPE.numpy(), MSE.numpy(),
                  RelativeMAE.numpy(), RelativeMSE.numpy()]
    

#%%
print(100*'*')
print('CGPGNN:')
print(Metrics_CGPGNN)
#%%
parameters = [p for p in trainer_CGP_GNN.model.parameters()]
parameters_name= [ name for (name, param) in CGP_GNN.named_parameters()]

#%%
elapsed = time.time() - t
print('Elapsed: %s' % round(elapsed/60,2), ' minutes')
print(600*'*')

print("To view Weights & Biases results, visit https://wandb.ai and check your project 'FINAL_MetrLA_M6_H3'")


