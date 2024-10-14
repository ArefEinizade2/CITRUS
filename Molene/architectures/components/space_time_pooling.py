import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils.graph_utils import computeNeighborhood, build_time_graph, build_parametric_product_graph


def build_support(n_timesteps, S_spatial, cyclic):
    if n_timesteps > 1:
        # TODO: do we want to make the directness of the pooling support flexible?
        time_support = build_time_graph(n_timesteps, directed=True, cyclic=cyclic)
        graph_time_support = build_parametric_product_graph(time_support, S_spatial, h_00=0, h_01=1, h_10=1, h_11=0)
    else:
        graph_time_support = torch.from_numpy(S_spatial)
    return graph_time_support

class SpaceTimeMaxPooling(nn.Module):
    """
    MaxPoolLocal Creates a pooling layer on graphs by selecting nodes

    Initialization:

        MaxPoolLocal(in_dim, out_dim, number_hops)

        Inputs:
            in_dim (int): number of nodes at the input
            out_dim (int): number of nodes at the output
            number_hops (int): number of hops to pool information

        Output:
            torch.nn.Module for a local max-pooling layer.

        Observation: The selected nodes for the output are always the top ones.

    Add a neighborhood set:

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before being used, we need to define the GSO
        that will determine the neighborhood that we are going to pool.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        v = MaxPoolLocal(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x in_dim

        Outputs:
            y (torch.tensor): pooled data; shape:
                batch_size x dim_features x out_dim
    """

    def __init__(self, S_spatial, cyclic: bool, is_time_directed: bool,
                 n_active_nodes_in, n_active_nodes_out,
                 n_timesteps_in, n_timesteps_out, n_hops, total_observations,
                 verbose:bool):

        super().__init__()
        self.verbose = verbose
        self.n_active_nodes_in = n_active_nodes_in
        self.n_active_nodes_out = n_active_nodes_out
        self.n_hops = n_hops
        self.n_timesteps_in = n_timesteps_in
        self.n_timesteps_out = n_timesteps_out
        self.n_nodes_to_keep_per_timestep = int(self.n_active_nodes_out / self.n_timesteps_out)
        self.total_observations = total_observations
        self.is_time_directed = is_time_directed
        self.S_spatial = S_spatial
        self.cyclic_time_graph = cyclic



        # we are performing pooling only on a cartesian product. It is not parametric.
        # if self.n_timesteps_in > 1:
        #
        #     self.S_time = build_time_graph(self.n_timesteps_in, directed=self.is_time_directed, cyclic=self.cyclic_time_graph)
        #     self.S_spacetime = build_parametric_product_graph(self.S_time, self.S_spatial, h_00=0, h_01=1, h_10=1, h_11=0)
        # else:
        #     self.S_spacetime = torch.from_numpy(S_spatial)
        self.S_spacetime = build_support(self.n_timesteps_in, self.S_spatial, cyclic=True)



        self.time_pooling_step = int(math.ceil(self.n_timesteps_in/self.n_timesteps_out))
        self.time_indices = [i for i in range(0, self.n_timesteps_in, self.time_pooling_step)]

        self.neighborhood = self.initialize_neighborhood()
        self.max_neighborhood_size = self.neighborhood.shape[1]


    def initialize_neighborhood(self):
        # nx.draw_networkx(nx.from_numpy_array(self.S_spacetime.numpy()), with_labels=True)
        # plt.show()

        neighbors = computeNeighborhood(S=self.S_spacetime.numpy(),
                                        K=self.n_hops,
                                        n_active_nodes_out='all',  # self.n_active_nodes_in,
                                        n_active_nodes_neighborhood='all', #self.n_active_nodes_in,
                                        outputType='matrix')
        neighbors = torch.tensor(neighbors)
        return neighbors

    @staticmethod
    def plot_signal_features(sample, support, prefix,
                             num_of_timesteps, total_nodes_per_timestep, active_nodes_per_timestep,
                             vmin=None, vmax=None):
        n_in_features = sample.shape[0]
        for feat_idx in range(n_in_features):
            if feat_idx != 1:
                continue
            graph_signal = sample[feat_idx]
            if graph_signal.shape[0] < support.shape[0]:
                # needs padding
                zero_padded_signal = -0.5 * torch.ones([num_of_timesteps, total_nodes_per_timestep])
                zero_padded_signal[:, :active_nodes_per_timestep] = graph_signal.reshape(num_of_timesteps, -1)
                padded_graph_signal = zero_padded_signal.reshape(1, -1).squeeze()
                graph_signal = padded_graph_signal
                vmin = -0.5

            nx.draw_networkx(nx.from_numpy_array(support.numpy()), with_labels=True,
                             node_color=graph_signal.tolist(),
                             vmin=vmin, vmax=vmax)
                             #vmin=np.min(graph_signal.tolist()), vmax=np.max(graph_signal.tolist()))
            plt.title(f"{prefix}: Feature {feat_idx + 1} of {n_in_features}")
            plt.show()

    def forward(self, x):
        # x should be of shape batch_size x n_feat x n_active_nodes_in
        PLOT = False
        sample_index = 0

        if PLOT:
            graph_time_support = build_support(self.n_timesteps_in, self.S_spatial, cyclic=False)
            self.plot_signal_features(x[sample_index], graph_time_support, "Input to pooling layer",
                                      num_of_timesteps=self.n_timesteps_in, total_nodes_per_timestep=self.S_spatial.shape[0],
                                      active_nodes_per_timestep=int(self.n_active_nodes_in/self.n_timesteps_in),
                                      vmin=0, vmax=5)

        batch_size = x.shape[0]
        n_feat = x.shape[1]
        assert x.shape[2] == self.n_active_nodes_in and x.shape[2] >= self.n_active_nodes_out

        n_nodes_support = self.S_spatial.shape[0] * self.n_timesteps_in
        nodes_to_keep_per_timestep = int(self.n_active_nodes_in / self.n_timesteps_in)

        if x.shape[2] < n_nodes_support:
            # need to zero-pad the 'x' to match the neighborhood support
            zero_padded_signal = torch.zeros([x.shape[0], x.shape[1], self.n_timesteps_in, self.S_spatial.shape[0]]).to(x.device)
            zero_padded_signal[:, :, :, :nodes_to_keep_per_timestep] = x.reshape(x.shape[0], x.shape[1], self.n_timesteps_in, -1)
            x = zero_padded_signal.reshape(x.shape[0], x.shape[1], -1)


        x = x.unsqueeze(3)  # B x F x N x 1
        x = x.repeat([1, 1, 1, self.max_neighborhood_size])  # B x F x N x maxNeighbor

        gatherNeighbor = self.neighborhood.unsqueeze(0).unsqueeze(0)
        gatherNeighbor = gatherNeighbor.repeat([batch_size, n_feat, 1, 1]).to(x.device)

        xNeighbors = torch.gather(x, 2, gatherNeighbor.type(torch.int64))

        x_summarized, _ = torch.max(xNeighbors, dim=3)  # [batch_size x features x (active_nodes_per_timestep*n_timesteps_in)]
        if self.n_active_nodes_in < n_nodes_support:
            # we have to remove the padded nodes
            x_summarized = x_summarized.reshape(x_summarized.shape[0], x_summarized.shape[1], self.n_timesteps_in, -1)\
                [:, :, :, :nodes_to_keep_per_timestep]\
                .reshape(x_summarized.shape[0], x_summarized.shape[1], -1)


        assert x_summarized.shape[2] == self.n_active_nodes_in

        if PLOT:
            graph_time_support = build_support(self.n_timesteps_in, self.S_spatial, cyclic=False)
            self.plot_signal_features(x_summarized[sample_index], graph_time_support, "After summarization",
                                      num_of_timesteps=self.n_timesteps_in, total_nodes_per_timestep=self.S_spatial.shape[0],
                                      active_nodes_per_timestep=int(self.n_active_nodes_in/self.n_timesteps_in),
                                      vmin=0, vmax=5)

        # OLD VERSION BEFORE REFORMULATION
        # x_in_timesteps = x_summarized.reshape(x_summarized.shape[0], -1, self.n_timesteps_in)  # [batch_size, active_nodes_per_timestep * n_feat, n_timesteps_in]
        # assert x_in_timesteps.shape[1] == self.n_active_nodes_in / self.n_timesteps_in * n_feat
        # x_sliced = x_in_timesteps[:, :, self.time_indices]  # [batch_size, active_nodes_per_timestep * n_feat, n_timesteps_out]
        # x_summarized_sliced = x_sliced.reshape(x_summarized.shape[0], n_feat, -1)  # [batch_size x features x (active_nodes_per_timestep*n_timesteps_out)]
        #  x_pooled = x_summarized_sliced[:, :, :self.n_active_nodes_out]  # [batch_size x features x nodes_out]

        # NEW VERSION AFTER REFORMULATION
        assert x_summarized.shape[2] % self.n_timesteps_in == 0  # verify that we can represent the input over timesteps
        chunk_size = int(x_summarized.shape[2]/self.n_timesteps_in)
        indices_to_select = [i+slice_num*chunk_size for slice_num in self.time_indices for i in range(chunk_size)]
        # Equivalent of:
        # indices_to_select = []
        # for slice_num in self.time_indices:
        #     for i in range(chunk_size):
        #         indices_to_select.append(i + slice_num * chunk_size)

        x_summarized_sliced = torch.index_select(x_summarized, dim=2, index=torch.from_numpy(np.array(indices_to_select)).long().to(x.device))
        assert x_summarized_sliced.shape[2] == self.n_active_nodes_in / self.n_timesteps_in * self.n_timesteps_out

        if PLOT:
            graph_time_support = build_support(self.n_timesteps_out, self.S_spatial, cyclic=False)
            self.plot_signal_features(x_summarized_sliced[sample_index], graph_time_support, f"After slicing (kept timesteps {self.time_indices})",
                                      num_of_timesteps=self.n_timesteps_out, total_nodes_per_timestep=self.S_spatial.shape[0],
                                      active_nodes_per_timestep=int(self.n_active_nodes_in/self.n_timesteps_in),
                                      vmin=0, vmax=5)

        x_downsampled = x_summarized_sliced.unsqueeze(2)\
            .reshape(batch_size, n_feat, self.n_timesteps_out, -1)[:, :, :, :self.n_nodes_to_keep_per_timestep]\
            .reshape(batch_size, n_feat, -1)
        # [batch x features x 1 x active_nodes_in]
        # [batch x features x timesteps_out x nodes_per_timestep_out] --> [batch x features x timesteps_out x selected_nodes]
        # [batch x features x 1 x active_nodes_out]
        assert x_downsampled.shape[1] == n_feat
        assert x_downsampled.shape[2] == self.n_active_nodes_out

        if PLOT and len(self.time_indices):
            graph_time_support = build_support(self.n_timesteps_out, self.S_spatial, cyclic=False)
            self.plot_signal_features(x_downsampled[sample_index], graph_time_support, f"Downsampled. Kept {self.n_nodes_to_keep_per_timestep} nodes per timestep",
                                      num_of_timesteps=self.n_timesteps_out, total_nodes_per_timestep=self.S_spatial.shape[0],
                                      active_nodes_per_timestep=int(self.n_active_nodes_out/self.n_timesteps_out),
                                      vmin=0, vmax=5)

        return x_downsampled

    def extra_repr(self):
        repr_string = f"nodes_in={self.n_active_nodes_in}, " \
            f"nodes_out={self.n_active_nodes_out}, " \
            f"hops={self.n_hops}, " \
            f"steps_in={self.n_timesteps_in}, " \
            f"steps_out={self.n_timesteps_out}, " \
            f"tot_st_nodes={self.S_spacetime.shape[0]}, " \
            f"time_pooling_step= {self.time_pooling_step}, " \
            f"slices_to_pool={self.time_indices}"
        return repr_string
