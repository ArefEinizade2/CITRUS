import torch


def LSIGF(h, S, x, b):
    # TODO: think about how to make this sparse
    """
    Taken from Fernando Gama's repository: https://github.com/alelab-upenn/graph-neural-networks

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
        h (torch.tensor): array of filter taps;
            shape:                                 output_features x edge_features x filter_taps x input_features
        S (torch.tensor): graph shift operator;
            shape:                                 edge_features x number_nodes x number_nodes
        x (torch.tensor): input signal;
            shape:                                 batch_size x input_features x number_nodes
        b (torch.tensor): shape:                   output_features x number_nodes
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

    # h is [output_features x edge_weights x filter_taps x input_features]
    # S is [edge_weighs x number_nodes x number_nodes]
    # x is [batch_size x input_features x number_nodes]
    # b is [output_features x number_nodes]

    # Output:
    # y is [batch_size x output_features x number_nodes]

    # Get the parameter numbers:
    num_feat_out = h.shape[0]
    num_edge_feat = h.shape[1]
    num_filter_taps = h.shape[2]
    num_feat_in = h.shape[3]

    # S is [edge_features x number_nodes x number_nodes]
    assert S.shape[0] == num_edge_feat
    n_nodes = S.shape[1]
    assert S.shape[2] == n_nodes

    batch_size = x.shape[0]
    assert x.shape[1] == num_feat_in
    assert x.shape[2] == n_nodes

    # Or, in the notation we've been using:
    # h in [F x E x K x G]
    # S in [E x N x N]
    # x in [B x G x N]
    # b in [F x N]
    # y in [B x F x N]

    # Now, we have:
    # - x [B x G x N]
    # - S [E x N x N]

    # E is often 1

    # and we want to come up with matrix multiplication that yields z = x * S with shape [B x E x K x G x N]

    # We will make:
    # - x [B x 1 x G x N]
    # - S [1 x E x N x N]

    # For this, we first add the corresponding dimensions
    x = x.reshape([batch_size, 1, num_feat_in, n_nodes])
    S = S.reshape([1, num_edge_feat, n_nodes, n_nodes])

    z = x.reshape([batch_size, 1, 1, num_feat_in, n_nodes])
    z = z.repeat(1, num_edge_feat, 1, 1, 1)

    # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    # now z is [B x E x 1 x G x N]

    for k in range(1, num_filter_taps):
        # - x [B x 1 x G x N]
        # - S [1 x 1 x N x N]
        x = torch.matmul(x, S)  # B x E x G x N

        xS = x.reshape([batch_size, num_edge_feat, 1, num_feat_in, n_nodes])  # B x 1 x 1 x G x N
        z = torch.cat((z, xS), dim=2)  # B x E x k x G x N
    # This output z is of size B x E x K x G x N

    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(
        z.permute(0, 4, 1, 2, 3).reshape([batch_size, n_nodes, num_edge_feat * num_filter_taps * num_feat_in]),
        h.reshape([num_feat_out, num_edge_feat * num_filter_taps * num_feat_in]).permute(1, 0))\
        .permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y










def SparseLSIGF(h, S, x, b):
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is [output_features x edge_weights x filter_taps x input_features]
    # S is [edge_weighs x number_nodes x number_nodes]
    # x is [batch_size x input_features x number_nodes]
    # b is [output_features x number_nodes]

    # Output:
    # y is [batch_size x output_features x number_nodes]

    # Get the parameter numbers:
    num_feat_out = h.shape[0]
    num_filter_taps = h.shape[2]
    num_feat_in = h.shape[3]

    # NEW
    h = h[:, 0, :, :]  # NEW
    S = S[0, :, :]  # NEW

    # S is [edge_features x number_nodes x number_nodes]
    assert S.shape[0] == S.shape[1]


    batch_size = x.shape[0]
    assert x.shape[1] == num_feat_in
    assert x.shape[2] == S.shape[0]
    n_nodes = S.shape[0]


    S = S.to_sparse()
    # Or, in the notation we've been using:
    # h in [F x E x K x G]  -> [F x K x G]
    # S in [E x N x N] -> [N x N]
    # x in [B x G x N]
    # b in [F x N]
    # y in [B x F x N]

    # Now, we have:
    # - x [B x G x N]
    # - S [N x N]

    # and we want to come up with matrix multiplication that yields z = x * S with shape [B x K x G x N]

    # We will make:
    # - x [B x G x N]
    # - S [1 x N x N]

    # For this, we first add the corresponding dimensions
    x = x.reshape([batch_size, num_feat_in, n_nodes])
    #S = S.reshape([1, n_nodes, n_nodes]).to_sparse()

    z = x.reshape([batch_size, 1, num_feat_in, n_nodes]).to(x.device)

    # now z is [B x 1 x G x N]

    for k in range(1, num_filter_taps):
        # - x [B x G x N]
        # - S [1 x N x N]
        res = torch.zeros(x.shape).to(x.device)
        for batch_idx in range(x.shape[0]):
            res[batch_idx] = torch.mm(S, x[batch_idx].transpose(dim0=0, dim1=1).to(x.device)).transpose(dim0=0, dim1=1)  # G x N
        #x = [B x G x N]
        x = res

        xS = x.reshape([batch_size, 1, num_feat_in, n_nodes])  # B x 1 x G x N
        z = torch.cat((z, xS), dim=1)  # B x K x G x N
    # This output z is of size B x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x K x G and reshape it to B x N x KG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    final_z = z.permute(0, 3, 1, 2).reshape([batch_size, n_nodes, num_filter_taps * num_feat_in])
    final_h = h.reshape([num_feat_out, num_filter_taps * num_feat_in]).permute(1, 0)

    y = torch.zeros(x.shape[0], x.shape[2], num_feat_out).to(x.device)
    for batch_idx in range(x.shape[0]):
        y[batch_idx] = torch.mm(final_z[batch_idx], final_h)
    # y = torch.matmul(final_z, final_h).permute(0, 2, 1)  # And permute again to bring it from B x N x F to B x F x N.
    y = y.permute(0, 2, 1)
    # Finally, add the bias
    if b is not None:
        y = y + b

    return y
