import datetime
import json

import numpy as np
from itertools import product
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from parametric_pooling_net_ordering import ParametricNetWithPoolingOrdered
from difference_utils import perform_step_ahead_deltas, compute_iteration_rNMSE_with_deltas, \
    visualize_predictions, visualize_deltas
from train_utils import train_model_regression
from evaluation import MSELossWithSparsityRegularizer, rNMSELoss, compute_iteration_rNMSE
from pred_utils import get_device, transform_data_to_all_steps_prediction, get_name_string, get_NOAA_dataset
from misc_utils import check_create_folder
from layers import CPGNN_ST, CPGNN_ST_v2, CPGNN_ST_v3, CITRUS, SGPModel
import networkx as nx
from Utilsss import get_evcs_evals

torch.cuda.current_device()



# torch.manual_seed(123)
# np.random.seed(123)
# random.seed(123)

device = get_device(use_gpu=True)

ds_folder = ""
splits = [0.35, 0.15, 0.5]
obs_window = 10
DIFFERENCE = False

data, steps_ahead, weighted_adjacency = get_NOAA_dataset(
    ds_folder,
    splits=splits,
    obs_window=obs_window,
    differenced=DIFFERENCE,
)
N_spatial_nodes = weighted_adjacency.shape[0]
print(f"{N_spatial_nodes} nodes - {obs_window} observed timesteps - steps ahead: {steps_ahead}")










# Get data
trn_data, val_data, tst_data_deltas, trn_labels, val_labels, tst_labels_deltas = transform_data_to_all_steps_prediction(data, node_first=True, device=device)
trn_data = trn_data.float()
val_data = val_data.float()
tst_data_deltas = tst_data_deltas.float()
trn_labels = trn_labels.float()
val_labels = val_labels.float()
tst_labels_deltas = tst_labels_deltas.float()
print(trn_data.shape, val_data.shape, tst_data_deltas.shape)

# obtain one-step labels for the training
one_step_trn_labels = trn_labels[:, 0, :]  # [batch x step-ahead x nodes]
one_step_val_labels = val_labels[:, 0, :]
print(one_step_trn_labels.shape, one_step_val_labels.shape)


N_ITERATIONS = 10
num_epochs = 400
learning_rate = 0.01# 0.0005
weight_decay = 0.0025  # , 0.00001, 0]
batch_size = 256  # 32
patience = 100
factor = 0.9

not_learning_limit = 100


lambda_value = 0  # 0.00025, 0.0005, 0.001, 0.005, 0.01, 0.05]


# GTCNN
n_nodes = weighted_adjacency.shape[0]
M = obs_window
N = [n_nodes, M]
k = list(np.array(N)-2)
K_list = [2, 2]

adj = weighted_adjacency
Graph_List = [nx.from_numpy_array(np.array(adj)), nx.path_graph(obs_window)]

evecs, evals, L_list = get_evcs_evals(Graph_List, K_list)

for ii in range(len(evals)):
    evals[ii] = evals[ii].to(device)
    
dim = 4
N_block = 3

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

res_dict = {
        'lr': learning_rate,
        'results': []
    }





for i in range(N_ITERATIONS):
    print(100*'*' + ' iter: ' + str(i) + 100*'*')
    one_step_gtcnn = CITRUS(input_size=1,
                                n_nodes=n_nodes,
                                horizon=1,
                                emb_size=dim,
                                hidden_size=dim,
                                rnn_layers=1,
                                gnn_kernel=1,
                                mass = torch.ones(np.prod(N)).to(device),
                                evals = evals,
                                evecs = torch.tensor(evecs).to(device),
                                C_width = dim,
                                N_block = N_block,
                                single_t = False,
                                use_gdc = [],
                                num_nodes = N,
                                last_activation=torch.nn.LeakyReLU(), 
                                mlp_hidden_dims=[dim, dim, dim, dim], 
                                dropout=False, 
                                with_MLP=True, 
                                diffusion_method='spectral', 
                                device = device,
                                graph_wise=False).to(device)
    one_step_gtcnn.to(device)
    print(one_step_gtcnn)

    model_parameters = filter(lambda p: p.requires_grad, one_step_gtcnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {params}")

    log_dir = f"./runs_NOAA_w={obs_window}/{today}_lr={learning_rate}_b={batch_size}_CPGNN_dim={dim}_l={N_block}_eig1={K_list[0]}_eig2={K_list[1]}"

    check_create_folder(log_dir)

    ### TRAINING ###
    # loss_criterion = rNMSELossWithSparsityRegularizer(one_step_gtcnn, lambda_reg)
    loss_criterion = MSELossWithSparsityRegularizer(one_step_gtcnn, lambda_value)

    val_metric = rNMSELoss()

    optimizer = torch.optim.Adam(one_step_gtcnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)

    best_model, best_epoch = train_model_regression(Iter=i,
        model=one_step_gtcnn,
        training_data=trn_data, validation_data=val_data,  # [n_samples x 1 x nodes x timesteps]
        single_step_trn_labels=one_step_trn_labels, single_step_val_labels=one_step_val_labels,  # [n_samples x spatial_nodes]
        num_epochs=num_epochs, batch_size=batch_size,
        loss_criterion=loss_criterion, optimizer=optimizer, scheduler=scheduler,
        val_metric_criterion=val_metric,
        log_dir=log_dir,
        not_learning_limit=not_learning_limit
    )

    rNMSE_dict, predictions_dict = compute_iteration_rNMSE(best_model, steps_ahead, tst_data_deltas, tst_labels_deltas,
                                                           device, verbose=False)

    res_dict['results'].append([round(l.item(), 4) for l in list(rNMSE_dict.values())])

    means = [round(el, 4) for el in np.average(res_dict['results'], axis=0)]
    stds = [round(el, 4) for el in np.std(res_dict['results'], axis=0)]
    res_dict['final_res'] = {
        'avg': means,
        'std': stds
    }

    with open(log_dir + '/results.json', 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    print(res_dict['results'])


print(res_dict['results'])
means = [round(el, 4) for el in np.average(res_dict['results'], axis=0)]
stds = [round(el, 4) for el in np.std(res_dict['results'], axis=0)]
print(means)
print(stds)

res_dict['final_res'] = {
    'avg': means,
    'std': stds
}
with open(log_dir + '/results.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict, f, ensure_ascii=False, indent=4)
    
    
    
