import math
import torch
from torch import nn
import numpy as np
from architectures.components.parametric_graph_filter import ParametricGraphFilter
from architectures.components.space_time_pooling import SpaceTimeMaxPooling
from utils.graph_utils import permutation_by_degree


class ParametricNetWithPoolingOrdered(torch.nn.Module):
    def __init__(self,
                 window: int, cyclic_time_graph: bool, time_directed: bool,

                 S_spatial: np.array,

                 n_feat_per_layer: list, n_taps_per_layer: list,

                 n_active_nodes_per_timestep_per_layer: list, time_pooling_ratio_per_layer: list,
                 pool_reach_per_layer: list,

                 output_dim: int,

                 device: str,

                 verbose: bool = False
                 ):
        """
        """
        super(ParametricNetWithPoolingOrdered, self).__init__()

        self.verbose = verbose
        if self.verbose:
            print("\n\n[ParametricNetWithPooling]. Initialization started.")
            print(f"Window is: {window}")
            print(f"N. nodes in spatial graph: {S_spatial.shape[0]}")
        self.window = window
        self.cyclic_time_graph = cyclic_time_graph
        self.is_time_directed = time_directed

        S_spatial_permuted, order = permutation_by_degree(S_spatial)
        self.S_spatial = S_spatial_permuted
        self.order = order
        self.reorder = [order.index(idx) for idx in list(range(len(order)))]

        self.n_feat_per_layer = n_feat_per_layer
        self.n_taps_per_layer = n_taps_per_layer
        self.n_active_nodes_per_timestep_per_layer = n_active_nodes_per_timestep_per_layer
        self.time_pooling_ratio_per_layer = time_pooling_ratio_per_layer
        self.pool_reach_per_layer = pool_reach_per_layer
        self.output_dim = output_dim
        self.device = device

        self.n_timesteps_per_layer = self.compute_timesteps_per_layer()
        self.n_active_nodes_at_each_layer = self.compute_active_nodes_per_layer()

        self.perform_dimensionality_checks()

        sequential_modules = self.build_layers()
        self.GFL = nn.Sequential(*sequential_modules)

        # Fully connected layer
        fc_in = self.n_active_nodes_at_each_layer[-1] * self.n_feat_per_layer[-1]
        fc_out = self.output_dim
        self.fc = nn.Linear(fc_in, fc_out)
        self.fc2 = nn.Linear(1, 10)

        if self.verbose:
            print("[ParametricNetWithPooling]. Initialization completed.")

    def compute_timesteps_per_layer(self):
        timesteps = [self.window]
        number_of_observations = self.window
        for pooling_factor in self.time_pooling_ratio_per_layer:
            pooling_factor = pooling_factor if pooling_factor <= number_of_observations else 1
            number_of_observations = math.ceil(number_of_observations / pooling_factor)
            timesteps.append(number_of_observations)

        if self.verbose:
            print(f"Timesteps per layer: {timesteps}")
        return timesteps

    def compute_active_nodes_per_layer(self):
        if self.verbose:
            print(f"N. active nodes per timestep per layer: {self.n_active_nodes_per_timestep_per_layer}")
        active_nodes_per_layer = []
        for i in range(len(self.n_active_nodes_per_timestep_per_layer)):
            actives_nodes = self.n_active_nodes_per_timestep_per_layer[i] * self.n_timesteps_per_layer[i]
            active_nodes_per_layer.append(actives_nodes)

        if self.verbose:
            print(f"N. of active nodes per layer: {active_nodes_per_layer}")
        return active_nodes_per_layer

    def perform_dimensionality_checks(self):
        n_layers = len(self.n_taps_per_layer)
        assert len(self.n_feat_per_layer) == n_layers + 1
        assert len(self.n_active_nodes_per_timestep_per_layer) == n_layers + 1
        assert len(self.n_taps_per_layer) == n_layers
        assert len(self.time_pooling_ratio_per_layer) == n_layers
        assert len(self.pool_reach_per_layer) == n_layers
        assert len(self.n_active_nodes_at_each_layer) == n_layers + 1
        assert len(self.S_spatial.shape) == 2 and self.S_spatial.shape[0] == self.S_spatial.shape[1]

    def build_layers(self):
        layers = []
        num_of_layers = len(self.n_taps_per_layer)
        for l in range(num_of_layers):
            param_filter = ParametricGraphFilter(S_spatial=self.S_spatial,
                                                 n_timesteps=self.n_timesteps_per_layer[l],
                                                 cyclic=self.cyclic_time_graph,
                                                 is_time_directed=self.is_time_directed,
                                                 n_feat_in=self.n_feat_per_layer[l],
                                                 n_feat_out=self.n_feat_per_layer[l + 1],
                                                 num_filter_taps=self.n_taps_per_layer[l],
                                                 device=self.device,
                                                 verbose=self.verbose)
            layers.append(param_filter)
            layers.append(torch.nn.ReLU())
            pooling = SpaceTimeMaxPooling(S_spatial=self.S_spatial,
                                          cyclic=self.cyclic_time_graph,
                                          is_time_directed=self.is_time_directed,
                                          n_active_nodes_in=self.n_active_nodes_at_each_layer[l],
                                          n_active_nodes_out=self.n_active_nodes_at_each_layer[l + 1],
                                          n_timesteps_in=self.n_timesteps_per_layer[l], n_timesteps_out=self.n_timesteps_per_layer[l + 1],
                                          n_hops=self.pool_reach_per_layer[l], total_observations=self.window,
                                          verbose=self.verbose)
            layers.append(pooling)
        return layers

    def forward(self, x):
        """
        x is of shape [batch_size, input_features, n_of_nodes, n_of_timesteps]
        """
        # print(x.shape)
        assert len(x.shape) == 4
        assert x.shape[1] == self.n_feat_per_layer[0]
        assert x.shape[2] == self.S_spatial.shape[0]
        assert x.shape[3] == self.n_timesteps_per_layer[0]

        x_nodes_permuted = x[:, :, self.order, :]
        # x_graph_time = x_nodes_permuted.flatten(start_dim=2)  # VERSION BEFORE REFORMULATION
        x_graph_time = x_nodes_permuted.transpose(dim0=2, dim1=3).flatten(start_dim=2)  # VERSION AFTER REFORMULATION
        # This operation performs the math VEC operation over the matrix [N x T] of the time-varying graph signal

        x_convoluted_pooled = self.GFL(x_graph_time)
        x_flattened = x_convoluted_pooled.reshape(x.shape[0], -1)  # flatten to feed into fc layer
        y = self.fc(x_flattened)

        if self.output_dim == 1:
            # binary classification
            y = torch.sigmoid(y)

        if y.shape[1] == self.S_spatial.shape[0]:
            # print("Reordering before output")
            y = y[:, self.reorder]
        y2 = self.fc2(torch.unsqueeze(y, -1))
        return y
