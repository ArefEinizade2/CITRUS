import math
import numpy as np
import torch
import torch.nn as nn


zeroTolerance = 1e-9  # Values below this number are considered zero.
infiniteNumber = 1e12  # infinity equals this number


def LSIGF_DB(h, S, x, b=None):
    """
    LSIGF_DB(filter_taps, GSO, input, bias=None) Computes the output of a
        linear shift-invariant graph filter (graph convolution) on delayed
        input and then adds bias.
    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e}(t) in R^{N x N} the GSO for edge feature e at time t,
    x(t) in R^{G x N} the input data at time t where x_{g}(t) in R^{N} is the
    graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.
    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}(t)S_{e}(t-1)...S_{e}(t-(k-1))
                                        x_{g}(t-k)
                + b_{f}
    for f = 1, ..., F.
    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x time_samples x edge_features
                                                  x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x time_samples x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}
    Outputs:
        output: filtered signals; shape:
            batch_size x time_samples x output_features x number_nodes
    """
    # This is the LSIGF with (B)atch and (D)elay capabilities (i.e. there is
    # a different GSO for each element in the batch, and it handles time
    # sequences, both in the GSO as in the input signal. The GSO should be
    # transparent).

    # So, the input
    #   h: F x E x K x G
    #   S: B x T x E x N x N
    #   x: B x T x G x N
    #   b: F x N
    # And the output has to be
    #   y: B x T x F x N

    # Check dimensions
    assert len(h.shape) == 4
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert len(S.shape) == 5
    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N
    assert len(x.shape) == 4
    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == G
    assert x.shape[3] == N

    # We would like a z of shape B x T x K x E x G x N that represents, for
    # each t, x_t, S_t x_{t-1}, S_t S_{t-1} x_{t-2}, ...,
    #         S_{t} ... S_{t-(k-1)} x_{t-k}, ..., S_{t} S_{t-1} ... x_{t-(K-1)}
    # But we don't want to do "for each t". We just want to do "for each k".

    # Let's start by reshaping x so it can be multiplied by S
    x = x.reshape([B, T, 1, G, N]).repeat(1, 1, E, 1, 1)

    # Now, for the first value of k, we just have the same signal
    z = x.reshape([B, T, 1, E, G, N])
    # For k = 0, k is counted in dim = 2

    # Now we need to start multiplying with S, but displacing the entire thing
    # once across time
    for k in range(1, K):
        # Across dim = 1 we need to "displace the dimension down", i.e. where
        # it used to be t = 1 we now need it to be t=0 and so on. For t=0
        # we add a "row" of zeros.
        x, _ = torch.split(x, [T - 1, 1], dim=1)
        #   The second part is the most recent time instant which we do not need
        #   anymore (it's used only once for the first value of K)
        # Now, we need to add a "row" of zeros at the beginning (for t = 0)
        zeroRow = torch.zeros(B, 1, E, G, N, dtype=x.dtype, device=x.device)
        x = torch.cat((zeroRow, x), dim=1)
        # And now we multiply with S
        x = torch.matmul(x, S)
        # Add the dimension along K
        xS = x.reshape(B, T, 1, E, G, N)
        # And concatenate it with z
        z = torch.cat((z, xS), dim=2)

    # Now, we finally made it to a vector z of shape B x T x K x E x G x N
    # To finally multiply with the filter taps, we need to swap the sizes
    # and reshape
    z = z.permute(0, 1, 5, 3, 2, 4)  # B x T x N x E x K x G
    z = z.reshape(B, T, N, E * K * G)
    # And the same with the filter taps
    h = h.reshape(F, E * K * G)
    h = h.permute(1, 0)  # E*K*G x F

    # Multiply
    y = torch.matmul(z, h)  # B x T x N x F
    # And permute
    y = y.permute(0, 1, 3, 2)  # B x T x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b

    return y


def GRNN_DB(a, b, S, x, z0, sigma,
            xBias=None, zBias=None):
    """
    GRNN_DB(signal_to_hidden_taps, hidden_to_hidden_taps, GSO, input,
            initial_hidden, nonlinearity, signal_bias, hidden_bias)
    Computes the sequence of hidden states for the input sequence x following
    the equation z_{t} = sigma(A(S) x_{t} + B(S) z_{t-1}) with initial state z0
    and where sigma is the nonlinearity, and A(S) and B(S) are the
    Input-to-Hidden filters and the Hidden-to-Hidden filters with the
    corresponding taps.

    Inputs:
        signal_to_hidden_taps (torch.tensor): shape
            hidden_features x edge_features x filter_taps x signal_features
        hidden_to_hidden_taps (torch.tensor): shape
            hidden_features x edge_features x filter_taps x hidden_features
        GSO (torch.tensor): shape
            batch_size x time x edge_features x number_nodes x number_nodes
        input (torch.tensor): shape
            batch_size x time x signal_features x number_nodes
        initial_hidden: shape
            batch_size x hidden_features x number_nodes
        signal_bias (torch.tensor): shape
            1 x 1 x hidden_features x 1
        hidden_bias (torch.tensor): shape
            1 x 1 x hidden_features x 1

    Outputs:
        hidden_state: shape
            batch_size x time x hidden_features x number_nodes

    """
    # We will compute the hidden state for a delayed and batch data.

    # So, the input
    #   a: H x E x K x F (Input to Hidden filters)
    #   b: H x E x K x H (Hidden to Hidden filters)
    #   S: B x T x E x N x N (GSO)
    #   x: B x T x F x N (Input signal)
    #   z0: B x H x N (Initial state)
    #   xBias: 1 x 1 x H x 1 (bias on the Input to Hidden features)
    #   zBias: 1 x 1 x H x 1 (bias on the Hidden to Hidden features)
    # And the output has to be
    #   z: B x T x H x N (Hidden state signal)

    # Check dimensions
    H = a.shape[0]  # Number of hidden state features
    E = a.shape[1]  # Number of edge features
    K = a.shape[2]  # Number of filter taps
    F = a.shape[3]  # Number of input features
    assert b.shape[0] == H
    assert b.shape[1] == E
    assert b.shape[2] == K
    assert b.shape[3] == H
    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N
    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == F
    assert x.shape[3] == N
    assert z0.shape[0] == B
    assert z0.shape[1] == H
    assert z0.shape[2] == N

    # The application of A(S) x(t) doesn't change (it does not depend on z(t))
    Ax = LSIGF_DB(a, S, x, b=xBias)  # B x T x H x N
    # This is the filtered signal for all time instants.
    # This also doesn't split S, it only splits x.

    # The b parameters we will always need them in this shape
    b = b.unsqueeze(0).reshape(1, H, E * K * H)  # 1 x H x EKH
    # so that we can multiply them with the product Sz that should be of shape
    # B x EKH x N

    # We will also need a selection matrix that selects the first K-1 elements
    # out of the original K (to avoid torch.split and torch.index_select with
    # more than one index)
    CK = torch.eye(K - 1, device=S.device)  # (K-1) x (K-1)
    zeroRow = torch.zeros((1, K - 1), device=CK.device)
    CK = torch.cat((CK, zeroRow), dim=0)  # K x (K-1)
    # This matrix discards the last column when multiplying on the left
    CK = CK.reshape(1, 1, 1, K, K - 1)  # 1(B) x 1(E) x 1(H) x K x K-1

    # \\\ Now compute the first time instant

    # We just need to multiplicate z0 = z(-1) by b(0) to get z(0)
    #   Create the zeros that will multiply the values of b(1), b(2), ... b(K-1)
    #   since we only need b(0)
    zerosK = torch.zeros((B, K - 1, H, N), device=z0.device)
    #   Concatenate them after z
    zK = torch.cat((z0.unsqueeze(1), zerosK), dim=1)  # B x K x H x N
    #   Now we have a signal that has only the z(-1) and the rest are zeros, so
    #   now we can go ahead and multiply it by b. For this to happen, we need
    #   to reshape it as B x EKH x N, but since we are always reshaping the last
    #   dimensions we will bring EKH to the end, reshape, and then put them back
    #   in the middle.
    zK = zK.reshape(B, 1, K, H, N).repeat(1, E, 1, 1, 1)  # B x E x K x H x N
    zK = zK.permute(0, 4, 1, 2, 3).reshape(B, N, E * K * H).permute(0, 2, 1)
    #   B x EKH x N
    #   Finally, we can go ahead an multiply with b
    zt = torch.matmul(b, zK)  # B x H x N
    # Now that we have b(0) z(0) we can add the bias, if necessary
    if zBias is not None:
        zt = zt + zBias
    # And we need to add it to a(0)x(0) which is the first element of Ax in the
    # T dimension
    # Let's do a torch.index_select; not so sure a selection matrix isn't better
    a0x0 = torch.index_select(Ax, 1, torch.tensor(0, device=Ax.device)).reshape(B, H, N)
    #   B x H x N
    # Recall that a0x0 already has the bias, so now we just need to add up and
    # apply the nonlinearity
    zt = sigma(a0x0 + zt)  # B x H x N
    z = zt.unsqueeze(1)  # B x 1 x H x N
    zt = zt.unsqueeze(1)  # B x 1 x H x N
    # This is where we will keep track of the product Sz
    Sz = z0.reshape(B, 1, 1, H, N).repeat(1, 1, E, 1, 1)  # B x 1 x E x H x N

    # Starting now, we need to multiply this by S every time
    for t in range(1, T):
        if t < K:
            # Get the current time instant
            St = torch.index_select(S, 1, torch.tensor(t, device=S.device))
            #   B x 1 x E x N x N
            # We need to multiply this time instant by all the elements in Sz
            # now, and there are t of those
            St = St.repeat(1, t, 1, 1, 1)  # B x t x E x N x N
            # Multiply by the newly acquired St to do one more delay
            Sz = torch.matmul(Sz, St)  # B x t x E x H x N
            # Observe that these delays are backward: the last element in the
            # T dimension (dim = 1) is the latest element, this makes sense
            # since that is the element we want to multiply by the last element
            # in b.

            # Now that we have delayed, add the newest value (which requires
            # no delay)
            ztThis = zt.unsqueeze(2).repeat(1, 1, E, 1, 1)  # B x 1 x E x H x N
            Sz = torch.cat((ztThis, Sz), dim=1)  # B x (t+1) x E x H x N

            # Pad all those values that are not there yet (will multiply b
            # by zero)
            zeroRow = torch.zeros((B, K - (t + 1), E, H, N), device=Sz.device)
            SzPad = torch.cat((Sz, zeroRow), dim=1)  # B x K x E x H x N

            # Reshape and permute to adapt to multiplication with b (happens
            # outside the if)
            bSz = SzPad.permute(0, 4, 2, 1, 3).reshape(B, N, E * K * H)
        else:
            # Now, we have t>=K which means that Sz is of shape
            #   B x K x E x H x N
            # and thus is full, so we need to get rid of the last element in Sz
            # before adding the new element and multiplying by St.

            # We can always get rid of the last element by multiplying by a
            # K x (K-1) selection matrix. So we do that (first we need to
            # permute to have the dimensions ready for multiplication)
            Sz = Sz.permute(0, 2, 3, 4, 1)  # B x E x H x N x K
            Sz = torch.matmul(Sz, CK)  # B x E x H x N x (K-1)
            Sz = Sz.permute(0, 4, 1, 2, 3)  # B x (K-1) x E x H x N

            # Get the current time instant
            St = torch.index_select(S, 1, torch.tensor(t, device=S.device))
            #   B x 1 x E x N x N
            # We need to multiply this time instant by all the elements in Sz
            # now, and there are K-1 of those
            St = St.repeat(1, K - 1, 1, 1, 1)  # B x (K-1) x E x N x N
            # Multiply by the newly acquired St to do one more delay
            Sz = torch.matmul(Sz, St)  # B x (K-1) x E x H x N

            # Now that we have delayed, add the newest value (which requires
            # no delay)
            ztThis = zt.unsqueeze(2).repeat(1, 1, E, 1, 1)  # B x 1 x E x H x N
            Sz = torch.cat((ztThis, Sz), dim=1)  # B x K x E x H x N

            # Reshape and permute to adapt to multiplication with b (happens
            # outside the if)
            bSz = Sz.permute(0, 4, 2, 1, 3).reshape(B, N, E * K * H)

        # Get back to proper order
        bSz = bSz.permute(0, 2, 1)  # B x EKH x N
        #   And multiply with the coefficients
        Bzt = torch.matmul(b, bSz)  # B x H x N
        # Now that we have the Bz for this time instant, add the bias
        if zBias is not None:
            Bzt = Bzt + zBias
        # Get the corresponding value of Ax
        Axt = torch.index_select(Ax, 1, torch.tensor(t, device=Ax.device))
        Axt = Axt.reshape(B, H, N)
        # Sum and apply the nonlinearity
        zt = sigma(Axt + Bzt).unsqueeze(1)  # B x 1 x H x N
        z = torch.cat((z, zt), dim=1)  # B x (t+1) x H x N

    return z  # B x T x H x N




#############################################################################
#                                                                           #
#              LAYERS (Filtering batch-dependent time-varying)              #
#                                                                           #
#############################################################################

class GraphFilter_DB(nn.Module):
    """
    GraphFilter_DB Creates a (linear) layer that applies a graph filter (i.e. a
        graph convolution) considering batches of GSO and the corresponding
        time delays
    Initialization:
        GraphFilter_DB(in_features, out_features, filter_taps,
                       edge_features=1, bias=True)
        Inputs:
            in_features (int): number of input features (each feature is a
                graph signal)
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
        GraphFilter_DB.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).
        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                batch_size x time_samples x edge_features
                                                  x number_nodes x number_nodes
    Forward call:
        y = GraphFilter_DB(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x time_samples x in_features x number_nodes
        Outputs:
            y (torch.tensor): output; shape:
                batch_size x time_samples x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
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
        self.S = None  # No GSO assigned yet
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
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x time x dimInFeatures x numberNodesIn
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        # F = x.shape[2]
        assert x.shape[3] == self.N
        # Compute the filter output
        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        # u is of shape batchSize x time x dimOutFeatures x numberNodes
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class HiddenState_DB(nn.Module):
    """
    HiddenState_DB Creates the layer for computing the hidden state of a GRNN

    Initialization:

        HiddenState_DB(signal_features, hidden_features, filter_taps,
                       nonlinearity=torch.tanh, edge_features=1, bias=True)

        Inputs:
            signal_features (int): number of signal features
            hidden_features (int): number of hidden state features
            filter_taps (int): number of filter taps (both filters have the
                same number of filter taps)
            nonlinearity (torch function): nonlinearity applied when computing
                the hidden state
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after each
                filter

        Output:
            torch.nn.Module for a hidden state layer

        Observation: Input-to-Hidden Filter taps have shape
                hidden_features x edge_features x filter_taps x signal_features
            Hidden-to-Hidden FIlter taps have shape
                hidden_features x edge_features x filter_taps x hidden_features

    Add graph shift operator:
    HiddenState_DB.addGSO(GSO) Before applying the layer, we need to define
    the GSO that we are going to use. This allows to change the GSO while
    using the same filtering coefficients (as long as the number of edge
    features is the same; but the number of nodes can change).
    Inputs:
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x time_samples x edge_features
                                              x number_nodes x number_nodes

    Forward call:
        y = HiddenState_DB(x, z0)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x time_samples x signal_features x number_nodes
            z0 (torch.tensor): initial hidden state; shape:
                batch_size x hidden_features x number_nodes
        Outputs:
            y (torch.tensor): hidden state; shape:
                batch_size x time_samples x hidden_features x number_nodes
    """

    def __init__(self, F, H, K, nonlinearity=torch.tanh, E=1, bias=True):
        # Initialize parent:
        super().__init__()

        # Store the values (using the notation in the paper):
        self.F = F  # Input Features
        self.H = H  # Hidden Features
        self.K = K  # Filter taps
        self.E = E  # Number of edge features
        self.S = None
        self.bias = bias  # Boolean
        self.sigma = nonlinearity  # torch.nn.functional

        # Create parameters:
        self.aWeights = nn.parameter.Parameter(torch.Tensor(H, E, K, F))
        self.bWeights = nn.parameter.Parameter(torch.Tensor(H, E, K, H))
        if self.bias:
            self.xBias = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.zBias = nn.parameter.Parameter(torch.Tensor(H, 1))
        else:
            self.register_parameter('xBias', None)
            self.register_parameter('zBias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.F * self.K)
        self.aWeights.data.uniform_(-stdv, stdv)
        self.bWeights.data.uniform_(-stdv, stdv)
        if self.bias:
            self.xBias.data.uniform_(-stdv, stdv)
            self.zBias.data.uniform_(-stdv, stdv)

    def forward(self, x, z0):

        assert self.S is not None

        # Input
        #   S: B x T (x E) x N x N
        #   x: B x T x F x N
        #   z0: B x H x N
        # Output
        #   z: B x T x H x N

        # Check dimensions
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        assert x.shape[2] == self.F
        N = x.shape[3]

        assert len(z0.shape) == 3
        assert z0.shape[0] == B
        assert z0.shape[1] == self.H
        assert z0.shape[2] == N

        z = GRNN_DB(self.aWeights, self.bWeights,
                    self.S, x, z0, self.sigma,
                    xBias=self.xBias, zBias=self.zBias)

        zT = torch.index_select(z, 1, torch.tensor(T - 1, device=z.device))
        # Give out the last one, to be used as starting point if used in
        # succession

        return z, zT.unsqueeze(1)

    def addGSO(self, S):
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def extra_repr(self):
        reprString = "in_features=%d, hidden_features=%d, " % (
            self.F, self.H) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias) + \
                     "nonlinearity=%s" % (self.sigma)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString