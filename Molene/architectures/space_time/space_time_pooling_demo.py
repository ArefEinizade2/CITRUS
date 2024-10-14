import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from architectures.space_time.parametric_pooling_net import ParametricNetWithPooling
from utils.data_utils import convert_src_loc_signals_to_product_graph, permute_src_loc_data
from utils.graph_utils import build_time_graph, permutation_by_degree, build_parametric_product_graph

use_gpu = True
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
print("Device selected: %s" % device)

# dummy data creation
obs_window = 4  # timesteps
batch_size = 100
S_time = build_time_graph(obs_window, directed=True, plot_graph=False)
S_spatial = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
])
dataset = torch.zeros(batch_size, S_time.shape[0], S_spatial.shape[0])  # [batch x time x nodes]
dataset[0, 0, 3] = 3
dataset[0, 1, 0] = 5
dataset[0, 2, 2] = 5
dataset[0, 3, 3] = 3
dataset[0, 3, 0] = 5


dataset[1, :, 0] = 5

dataset[2, :, 0] = 3

sample_index = 1

dataset = convert_src_loc_signals_to_product_graph(dataset)

# graph stuff
perm_S_spatial, order = permutation_by_degree(S_spatial)

# permute the dataset
permuted_dataset = permute_src_loc_data(dataset, obs_window, order)
permuted_dataset = permuted_dataset.to(device)

multidimensional_dataset = torch.zeros(permuted_dataset.shape[0], 2, permuted_dataset.shape[2])
multidimensional_dataset[0, 0, :] = permuted_dataset[1, 0, :]
multidimensional_dataset[0, 1, :] = permuted_dataset[2, 0, :]
multidimensional_dataset = multidimensional_dataset.to(device)

PLOT = True
if PLOT:
    S_spacetime_before_permutation = build_parametric_product_graph(S_spatial, S_time, 0, 1, 1, 0)
    space_time_graph_before_permutation = nx.from_numpy_matrix(S_spacetime_before_permutation.numpy())
    sample_before_permutation = dataset[sample_index][0]
    plt.figure(figsize=(7, 5))
    nx.draw_networkx(space_time_graph_before_permutation, with_labels=True, node_color=sample_before_permutation.tolist(), vmin=-4, vmax=5)
    plt.show()

    S_spacetime = build_parametric_product_graph(perm_S_spatial, S_time, 0, 1, 1, 0)
    space_time_graph = nx.from_numpy_matrix(S_spacetime.numpy())
    sample = permuted_dataset[sample_index][0]
    plt.figure(figsize=(7, 5))
    nx.draw_networkx(space_time_graph, with_labels=True, node_color=sample.tolist(), vmin=-4, vmax=5)
    plt.show()

# architecture
model = ParametricNetWithPooling(
    window=obs_window,
    S_spatial=perm_S_spatial,
    n_feat_per_layer=[2, 2, 3, 5],
    n_taps_per_layer=[3, 3, 3],
    n_active_nodes_per_timestep_per_layer=[5, 5, 3, 2],
    time_pooling_ratio_per_layer=[2, 2, 2],
    pool_reach_per_layer=[2, 2, 2],
    output_dim=2,
    device=device
)

print(model)
model.to(device)

#
out = model(permuted_dataset)
print(out.shape)
