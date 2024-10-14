import math
import torch
from scipy.sparse import kron
from torch import nn
from torch.nn import Parameter
import numpy as np

from architectures.components.lsigf import LSIGF
from utils.graph_utils import build_time_graph


class ParametricGraphFilter(nn.Module):
    def __init__(self, S_spatial, n_timesteps, cyclic: bool, is_time_directed: bool,
                 n_feat_in, n_feat_out, num_filter_taps,
                 device: str, verbose: bool):
        super(ParametricGraphFilter, self).__init__()

        self.verbose = verbose
        # Save parameters (notation of Fernando's paper):
        self.G = n_feat_in
        self.F = n_feat_out
        self.K = num_filter_taps
        self.E = 1  # we assume we do not have edge features
        self.cyclic_time_graph = cyclic

        self.n_timesteps = n_timesteps
        self.is_time_directed = is_time_directed
        self.S_spatial = S_spatial
        assert len(S_spatial.shape) == 2
        assert S_spatial.shape[0] == S_spatial.shape[1]


        if self.n_timesteps > 1:
            self.S_time = build_time_graph(n_timesteps, directed=self.is_time_directed, cyclic=self.cyclic_time_graph)
            assert len(self.S_time.shape) == 2
            assert self.S_time.shape[0] == self.S_time.shape[1]

        self.n_nodes_space_time_graph = S_spatial.shape[0] * (self.S_time.shape[1] if self.n_timesteps > 1 else 1)

        # The GSO has to be reconstructed at each forwards pass, since it is parametric.
        self.S = None

        # Create parameters (we assume there is bias):
        # noinspection PyArgumentList
        self.weights = Parameter(torch.Tensor(self.F, self.E, self.K, self.G))

        # noinspection PyArgumentList
        self.bias = Parameter(torch.Tensor(n_feat_out, 1))

        # Parameters for the parametric product graph
        self.S_spatial = torch.from_numpy(self.S_spatial).float().to(device)

        if self.n_timesteps > 1:
            # build parametric product graph components and initialize s_ij parameters
            self.S_0 = self.S_time
            self.S_1 = self.S_spatial

            self.I_0 = np.eye(self.S_0.shape[1])
            self.I_1 = np.eye(self.S_1.shape[1])
            self.S_kron_II = torch.from_numpy(kron(self.I_0, self.I_1).todense()).float().to(device)
            self.S_kron_SI = torch.from_numpy(kron(self.S_0, self.I_1).todense()).float().to(device)
            self.S_kron_IS = torch.from_numpy(kron(self.I_0, self.S_1.cpu().numpy()).todense()).float().to(device)
            self.S_kron_SS = torch.from_numpy(kron(self.S_0, self.S_1.cpu().numpy()).todense()).float().to(device)

            # if you want to learn the parameters, leave: Parameter(torch.ones(1))
            # if you want to fix the structure, choose:
            # - torch.zeros(1).float().to(device)
            # - torch.ones(1).float().to(device)
            self.s_00 = Parameter(torch.ones(1))
            self.s_01 = Parameter(torch.ones(1))
            self.s_10 = Parameter(torch.ones(1))
            self.s_11 = Parameter(torch.ones(1))

        # Initialize parameters
        self.initialize_weights_random()

        # For logging purposes
        self.n_parameters = self.weights.nelement() + self.bias.nelement() + 4

    def initialize_weights_xavier(self):
        gain = nn.init.calculate_gain('relu')
        print(f"Initialization (xavier) with gain {gain}")
        nn.init.xavier_uniform_(self.weights, gain=gain)
        self.bias.data.fill_(0.01)

    def initialize_weights_random(self):
        gain = 0.2
        print(f"Initialization (random) with gain {gain}")
        self.weights.data.uniform_(-gain, gain)
        self.bias.data.fill_(0.2)

    def compose_parametric_GSO(self):
        S = self.s_00 * self.S_kron_II + \
            self.s_01 * self.S_kron_IS + \
            self.s_10 * self.S_kron_SI + \
            self.s_11 * self.S_kron_SS
        return S

    def forward(self, x):
        """
        :param x: input [batch_size x num_feat_input x n_active_nodes_in]
        :return:
        """

        # compose parametric GSO for this forward pass
        if self.n_timesteps > 1:
            self.S = self.compose_parametric_GSO()
        else:
            self.S = self.S_spatial

        assert self.S.shape[0] == self.S.shape[1] == self.n_nodes_space_time_graph

        # reshape it to have [1 x n_nodes_space_time_graph x n_nodes_space_time_graph]
        self.S = self.S.reshape([1, self.S.shape[0], self.S.shape[1]])

        batch_size = x.shape[0]
        num_feat_input = x.shape[1]
        n_active_nodes_in = x.shape[2]


        if n_active_nodes_in < self.n_nodes_space_time_graph:
            # ZERO PADDING (NEW REFORMULATION)
            nodes_to_keep_per_timestep = int(n_active_nodes_in / self.n_timesteps)
            x_reshaped = x.reshape(x.shape[0], x.shape[1], self.n_timesteps, -1)

            zero_padded_signal = torch.zeros([x.shape[0], x.shape[1], self.n_timesteps, self.S_spatial.shape[0]])
            zero_padded_signal[:, :, :, :nodes_to_keep_per_timestep] = x_reshaped
            x = zero_padded_signal.reshape(x.shape[0], x.shape[1], -1).to(x.device)

            # OLD FORMULATION
            # zero-padding. This concatenates (self.n_nodes_space_time_graph - n_active_nodes_in) zeros to the input signal 'x'
            # along the third dimension (which represents the nodes).
            # x = torch.cat((x,
            #                torch.zeros(batch_size, num_feat_input, self.n_nodes_space_time_graph - n_active_nodes_in)
            #                .type(x.dtype).to(x.device)
            #                ), dim=2)

        # Compute the filter output
        u = LSIGF(self.weights, self.S, x, self.bias)

        # return only the values that are actually active
        if n_active_nodes_in < self.n_nodes_space_time_graph:
            # REVERSE ZERO-PADDING (NEW REFORMULATION)
            nodes_to_keep_per_timestep = int(n_active_nodes_in / self.n_timesteps)
            assert u.shape[2] % self.n_timesteps == 0  # verify that we can represent u over timesteps
            u = u.unsqueeze(2) \
                                .reshape(batch_size, u.shape[1], self.n_timesteps, -1)[:, :, :, :nodes_to_keep_per_timestep] \
                                .reshape(batch_size, u.shape[1], -1)

            assert u.shape[2] == n_active_nodes_in
            # OLD VERSION
            # u = torch.index_select(u, dim=2, index=torch.arange(n_active_nodes_in).to(u.device))

        return u

    def extra_repr(self):
        repr_string = f"in_features={self.G}, " \
            f"out_features={self.F}, " \
            f"filter_taps={self.K}, " \
            f"edge_features={self.E}, " \
            f"timesteps= {self.n_timesteps}"

        if self.n_timesteps > 1:
            repr_string += f", h_00= {self.s_00.cpu().item()}, " \
                f"h_01= {self.s_01.cpu().item()}, " \
                f"h_10= {self.s_10.cpu().item()}, " \
                f"h_11= {self.s_11.cpu().item()}"

        repr_string += f"\t# params= {self.n_parameters}"
        return repr_string
