import math
import numpy as np
import torch
import torch.nn as nn

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

class GGCRNNCell(nn.Module):
    """
    GGCRNNCell Creates a gated recurrent layer that computes h_t = sigma(\hat{Q_t}(A(S)*x_t)
    + \check{Q_t}(B(S)*h_{t-1})), where sigma is a nonlinearity (e.g. tanh), \hat{Q_t} and
    \check{Q_t} are the input and forget gate operators, A(S) and B(S) are LSI-GFs and h_t and x_t are the state and
    input variables respectively. \hat{Q_t} and \check{Q_t} can be time, node or edge gates (or time+node, time+edge).

    Initialization:

        GGCRNNCell(in_features, state_features, in_filter_taps,
                    state_filter_taps, sigma, time_gating=True, spatial_gating=None,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            state_features (int): number of state features (each feature is a
                graph signal)
            in_filter_taps (int): number of filter taps of the input filter
            state_filter_taps (int): number of filter taps of the state filter
            sigma (torch.nn): state nonlinearity (default tanh)
            time_gating (bool) = flag for time gating (default True)
            spatial_gating (string) = 'node' or 'edge' gating (default None)
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a gated graph recurrent layer.

        Observation: Input filter taps have shape
            state_features x edge_features x filter_taps x in_features

            State filter taps have shape
            state_features x edge_features x filter_taps x state_features

    Add graph shift operator:

        GGCRNNCell.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Here we also define the GCRNNs for the input and forget gates, as they
        use the same GSO

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        H = GGCRNNCell(X, h0)

        Inputs:
            X (torch.tensor): input data; shape:
                batch_size x sequence_length x in_features x number_nodes
            h0 (torch.tensor): initial hidden state; shape:
                batch_size x state_features x number_nodes

        Outputs:
            H (torch.tensor): output; shape:
                batch_size x sequence_length x state_features x number_nodes

    """

    def __init__(self, G, F, Kin, Kst, sigma=nn.Tanh, time_gating=True, spatial_gating=None, E=1, bias=True):
        # G: number of input features
        # F: number of state features
        # Kin, Kst: number of filter taps
        # sigma: nonlinearity (default tanh)
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.Kin = Kin
        self.Kst = Kst
        self.E = E
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight_A = nn.parameter.Parameter(torch.Tensor(F, E, Kin, G))
        self.weight_B = nn.parameter.Parameter(torch.Tensor(F, E, Kst, F))
        self.sigma = sigma
        self.time_gating = time_gating
        self.spatial_gating = spatial_gating
        self.bias_flag = bias

        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.Kin)
        self.weight_A.data.uniform_(-stdv, stdv)
        self.weight_B.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

        # If we want time gating, we have to define the GCRNN architectures of
        # the input, forget and out gates
        if self.time_gating == True:
            dimInputMLP = self.N * self.F  # Input dimension of fully connected layer needed for scalar gate computation
            # (i.e., we have N nodes, each one described by F features which
            #  means this will be flattened into a vector of size N*F

            # Architecture of input gate:
            # \\\ Graph filtering layers \\\
            GFL_in = GGCRNNCell(self.G, self.F, self.Kin, self.Kst, self.sigma, time_gating=False,
                                E=self.E, bias=self.bias_flag)  # Graph Filtering Layers
            GFL_in.addGSO(self.S)
            self.GFL_in = GFL_in
            # \\\ MLP (Fully Connected Layers) \\\
            fc_in = []
            fc_in.append(nn.Linear(dimInputMLP, 1, bias=self.bias_flag))
            fc_in.append(nn.Sigmoid())  # to ensure scalar between 0 and 1
            # And we're done
            self.MLP_in = nn.Sequential(*fc_in)
            # so we finally have the architecture for the input gate.

            # Architecture of forget gate:
            # \\\ Graph filtering layers \\\
            GFL_forget = GGCRNNCell(self.G, self.F, self.Kin, self.Kst,
                                    self.sigma, time_gating=False,
                                    E=self.E, bias=self.bias_flag)  # Graph Filtering Layers
            GFL_forget.addGSO(self.S)
            self.GFL_forget = GFL_forget
            # \\\ MLP (Fully Connected Layers) \\\
            fc_forget = []
            fc_forget.append(nn.Linear(dimInputMLP, 1, bias=self.bias_flag))
            fc_forget.append(nn.Sigmoid())  # to ensure scalar between 0 and 1
            self.MLP_forget = nn.Sequential(*fc_forget)

            # Architecture of output gate:
            # \\\ Graph filtering layers \\\
            GFL_out = GGCRNNCell(self.G, self.F, self.Kin, self.Kst, self.sigma,
                                 time_gating=False, E=self.E, bias=self.bias_flag)  # Graph Filtering Layers
            GFL_out.addGSO(self.S)
            self.GFL_out = GFL_out
            # \\\ MLP (Fully Connected Layers) \\\
            fc_out = []
            fc_out.append(nn.Linear(dimInputMLP, 1, bias=self.bias_flag))
            fc_out.append(nn.Sigmoid())  # to ensure scalar between 0 and 1
            self.MLP_out = nn.Sequential(*fc_out)

        if self.spatial_gating is not None:
            if self.spatial_gating == 'node':
                # Architecture of node gates:

                # \\\ Graph recurrent layers for input node gates \\\
                GRNN_node_in = GGCRNNCell(self.G, self.F, self.Kin, self.Kst, self.sigma,
                                          time_gating=False, E=self.E, bias=self.bias_flag)
                GRNN_node_in.addGSO(self.S)
                self.GRNN_node_in = GRNN_node_in
                # \\\  Graph Filtering Layers \\\
                gfl_node_in = []
                gni = GraphFilter(self.F, 1, self.Kst, self.E, self.bias_flag)
                gni.addGSO(self.S)
                gfl_node_in.append(gni)
                gfl_node_in.append(nn.Sigmoid())  # to ensure scalar between 0 and 1
                # And we're done
                self.GFL_node_in = nn.Sequential(*gfl_node_in)
                # so we finally have the architecture for the input gate.

                # \\\ Graph recurrent layers for state node gates \\\
                GRNN_node_forget = GGCRNNCell(self.G, self.F, self.Kin, self.Kst, self.sigma,
                                              time_gating=False, E=self.E, bias=self.bias_flag)
                GRNN_node_forget.addGSO(self.S)
                self.GRNN_node_forget = GRNN_node_forget
                # \\\  Graph Filtering Layers \\\
                gfl_node_forget = []
                gnf = GraphFilter(self.F, 1, self.Kst, self.E, self.bias_flag)
                gnf.addGSO(self.S)
                gfl_node_forget.append(gnf)
                gfl_node_forget.append(nn.Sigmoid())  # to ensure scalar between 0 and 1
                # And we're done
                self.GFL_node_forget = nn.Sequential(*gfl_node_forget)

            elif self.spatial_gating == 'edge':
                # Input edge gating \\
                input_attention = GraphAttentional(self.F, self.F, 1)
                input_attention.addGSO(self.S)
                self.input_attention = input_attention

                # State edge gating \\
                forget_attention = GraphAttentional(self.F, self.F, 1)
                forget_attention.addGSO(self.S)
                self.forget_attention = forget_attention

    def forward(self, X, h0):
        # X is of shape: batchSize x seqLen x dimInFeatures x numberNodesIn
        # h0 is of shape: batchSize x dimStFeatures x numberNodesIn
        assert h0.shape[0] == X.shape[0]
        B = X.shape[0]
        T = X.shape[1]
        F_in = X.shape[2]

        # Defaults when there is no time gating
        in_gate = torch.ones([B, 1]);
        forget_gate = torch.ones([B, 1]);

        # Compute the GCRNN cell output
        H = torch.empty(0).to(X.device)  # state sequence
        h = h0  # current state
        for i in range(T):
            # Slice input at time t
            x = torch.narrow(X, 1, i, 1)
            x = x.view(B, F_in, self.N)

            # Calculating time input and forget gates
            if self.time_gating == True:
                x_gates = x.view(B, 1, F_in, self.N)

                # Input gate
                # GRNN layer
                in_gate_state = self.GFL_in(x_gates, h0)
                # Flattening
                in_gate_state = in_gate_state.reshape(B, self.F * self.N)
                # Fully connected layer
                in_gate = self.MLP_in(in_gate_state)

                # Forget gate
                # GRNN layer
                forget_gate_state = self.GFL_forget(x_gates, h0)
                # Flattening
                forget_gate_state = forget_gate_state.reshape(B, self.F * self.N)
                # Fully connected layer
                forget_gate = self.MLP_forget(forget_gate_state)

            # Calculating node/edge gates
            if self.spatial_gating is not None:
                x_gates = x.view(B, 1, F_in, self.N)
                if self.spatial_gating == 'node':

                    # Input node gate
                    # GRNN layer
                    node_in_gate_state = self.GRNN_node_in(x_gates, h0)
                    # Flattening
                    node_in_gate_state = node_in_gate_state.reshape(B, self.F, self.N)
                    # GNN layer
                    node_in_gate = self.GFL_node_in(node_in_gate_state)
                    # Reshaping for elementwise multiplication
                    node_in_gate = node_in_gate.repeat(1, self.F, 1)

                    # Forget node gate
                    # GRNN layer
                    node_forget_gate_state = self.GRNN_node_forget(x_gates, h0)
                    # Flattening
                    node_forget_gate_state = node_forget_gate_state.reshape(B, self.F, self.N)
                    # GNN layer
                    node_forget_gate = self.GFL_node_forget(node_forget_gate_state)
                    # Reshaping for elementwise multiplication
                    node_forget_gate = node_forget_gate.repeat(1, self.F, 1)

                    # Apply filters to input and state
                    h = in_gate.view(-1, 1, 1).repeat(1, self.F, self.N) * (
                            node_in_gate * LSIGF(self.weight_A, self.S, x, self.bias)) \
                        + forget_gate.view(-1, 1, 1).repeat(1, self.F, self.N) * (
                                node_forget_gate * LSIGF(self.weight_B, self.S, h, self.bias))
                    # Apply hyperbolic tangent
                    h = self.sigma(h)

                elif self.spatial_gating == 'edge':
                    # Apply filters to input and state
                    h = in_gate.view(-1, 1, 1).repeat(1, self.F, self.N).to(x.device) * self.input_attention(
                        LSIGF(self.weight_A, self.S.to(x.device).float(), x, self.bias)) \
                        + forget_gate.view(-1, 1, 1).repeat(1, self.F, self.N).to(h.device) * self.forget_attention(
                        LSIGF(self.weight_B, self.S.to(h.device).float(), h, self.bias))
                    # Apply hyperbolic tangent
                    h = self.sigma(h)
            else:
                # Just time gating (if True)
                # Apply filters
                h = in_gate.view(-1, 1, 1).repeat(1, self.F, self.N).to(x.device) * LSIGF(self.weight_A, self.S.to(x.device), x, self.bias) + \
                    forget_gate.view(-1, 1, 1).repeat(1, self.F, self.N).to(h.device) * LSIGF(self.weight_B, self.S.to(h.device), h, self.bias)
                # Apply hyperbolic tangent
                h = self.sigma(h)

            h_unsq = h.unsqueeze(1)  # re-create sequence dimension
            # concatenate last state, h, with memory sequence H
            H = torch.cat([H, h_unsq], 1)
        return H


class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y



class GraphAttentional(nn.Module):
    """
    GraphAttentional Creates a graph attentional layer

    Initialization:

        GraphAttentional(in_features, out_features, attention_heads,
                         edge_features=1, nonlinearity=nn.functional.relu,
                         concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged.

        Output:
            torch.nn.Module for a graph attentional layer.

    Add graph shift operator:

        GraphAttentional.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1,
        nonlinearity = nn.functional.relu, concatenate = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(K, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(K, E, F, G))
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output
        y = graphAttention(x, self.mixer, self.weight, self.S)
        # This output is of size B x K x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x KF x N such that first iterates over f
            # and then over k: (k=0,f=0), (k=0,f=1), ..., (k=0,f=F-1), (k=1,f=0),
            # (k=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.K*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

        return y

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "attention_heads=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString



def graphAttention(x, a, W, S, negative_slope=0.2):
    """
    graphAttention(x, a, W, S) Computes attention following GAT layer taking
        into account multiple edge features.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ek} in R^{F x G} the weight marix associated to edge feature e and
    attention head k, and a^{ek} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ek} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head k, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Let y_{i}^{k} in R^{F} be the output of the graph attention at node i for
    attention head k. It is computed as
        y_{i}^{k} = \sum_{e=1}^{E}
                        \sum_{j in N_{i}}
                            s_{ij}^{e} alpha_{ij}^{ek} W^{ek} x_{j}
    with
        alpha_{ij}^{ek} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ek})^T [cat(W^{ek}x_{i}, W^{ek} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x: input; shape: batch_size x input_features x number_nodes
        a: mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W: linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S: graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        negative_slope: negative slope of the leaky relu

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    K = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == K
    assert W.shape[1] == E
    F = W.shape[2] # output_features
    assert a.shape[2] == int(2*F)
    G = W.shape[3] # input_features
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    S = (S + torch.eye(N).reshape([1,N,N]).repeat(E,1,1).to(S.device).double()).to(x.device)
    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size K x E x F x G
    # a is of size K x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, K, E, F, G])
    Wx = torch.matmul(W, x) # B x K x E x F x N
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x K x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device)) # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2*F).to(x.device)) # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, K, E, 1, F]), Wx) # B x K x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, K, E, 1, F]), Wx) # B x K x E x 1 x N
    # And then, use this to sum them accordingly and create a B x K x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3) # B x K x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    eij = nn.functional.leaky_relu(aWx, negative_slope = negative_slope)
    #   B x K x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim = 0)
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype)
    #   Make it -infinity where there are zeros
    infinityMask = (1-maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij*maskEdges - infinityMask, dim = 4)
    #   B x K x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    aij = aij * maskEdges # B x K x E x N x N
    # Finally, we just need to apply this matrix to the Wx which we have already
    # computed, and done.
    y = torch.matmul(Wx, S.reshape([1, 1, E, N, N]).float() * aij) # B x K x E x F x N
    # And sum over all edges
    return torch.sum(y, dim = 2) # B x K x F x N




