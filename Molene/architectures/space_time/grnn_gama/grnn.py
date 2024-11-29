import torch
import numpy as np
import torch.nn as nn

from architectures.space_time.grnn_gama.graphml import HiddenState_DB, GraphFilter_DB


class GraphRecurrentNN_DB(nn.Module):
    """
    GraphRecurrentNN_DB: implements the GRNN architecture on a time-varying GSO
        batch and delayed signals. It is a single-layer GRNN and the hidden
        state is initialized at random drawing from a standard gaussian.

    Initialization:

        GraphRecurrentNN_DB(dimInputSignals, dimOutputSignals,
                            dimHiddenSignals, nFilterTaps, bias, # Filtering
                            nonlinearityHidden, nonlinearityOutput,
                            nonlinearityReadout, # Nonlinearities
                            dimReadout, # Local readout layer
                            dimEdgeFeatures) # Structure

        Input:
            /** Graph convolutions **/
            dimInputSignals (int): dimension of the input signals
            dimOutputSignals (int): dimension of the output signals
            dimHiddenSignals (int): dimension of the hidden state
            nFilterTaps (list of int): a list with two elements, the first one
                is the number of filter taps for the filters in the hidden
                state equation, the second one is the number of filter taps
                for the filters in the output
            bias (bool): include bias after graph filter on every layer

            /** Activation functions **/
            nonlinearityHidden (torch.function): the nonlinearity to apply
                when computing the hidden state; it has to be a torch function,
                not a nn.Module
            nonlinearityOutput (torch.function): the nonlinearity to apply when
                computing the output signal; it has to be a torch function, not
                a nn.Module.
            nonlinearityReadout (nn.Module): the nonlinearity to apply at the
                end of the readout layer (if the readout layer has more than
                one layer); this one has to be a nn.Module, instead of just a
                torch function.

            /** Readout layer **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).

            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features

        Output:
            nn.Module with a GRNN architecture with the above specified
            characteristics that considers time-varying batch GSO and delayed
            signals

    Forward call:

        GraphRecurrentNN_DB(x, S)

        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimInputSignals x numberNodes
            GSO (torch.tensor): graph shift operator; shape
                batchSize x timeSamples (x dimEdgeFeatures)
                                                    x numberNodes x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the GRNN;
                batchSize x timeSamples x dimReadout[-1] x numberNodes

    Other methods:

        y, yGNN = .splitForward(x, S): gives the output of the entire GRNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of the GRNN (i.e. before the readout layers),
        yGNN of shape batchSize x timeSamples x dimInputSignals x numberNodes.
        This can be used to isolate the effect of the graph convolutions from
        the effect of the readout layer.

        y = .singleNodeForward(x, S, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize
        x timeSamples x dimInputSignals x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize
        x timeSamples x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimInputSignals,
                 dimOutputSignals,
                 dimHiddenSignals,
                 nFilterTaps, bias,
                 # Nonlinearities
                 nonlinearityHidden,
                 nonlinearityOutput,
                 nonlinearityReadout,  # nn.Module
                 # Local MLP in the end
                 dimReadout,
                 # Structure
                 dimEdgeFeatures,

                 #Global classification (added by Gabriele Mazzola)
                 n_classes=None,
                 n_nodes=None
                 ):
        # Initialize parent:
        super().__init__()

        # A list of two int, one for the number of filter taps (the computation
        # of the hidden state has the same number of filter taps)
        assert len(nFilterTaps) == 2

        # Store the values (using the notation in the paper):
        self.F = dimInputSignals  # Number of input features
        self.G = dimOutputSignals  # Number of output features
        self.H = dimHiddenSignals  # NUmber of hidden features
        self.K = nFilterTaps  # Filter taps
        self.E = dimEdgeFeatures  # Number of edge features
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearityHidden
        self.rho = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout
        self.dimReadout = dimReadout
        # \\\ Hidden State RNN \\\
        # Create the layer that generates the hidden state, and generate z0
        self.hiddenState = HiddenState_DB(self.F, self.H, self.K[0],
                                              nonlinearity=self.sigma, E=self.E,
                                              bias=self.bias)
        # \\\ Output Graph Filters \\\
        self.outputState = GraphFilter_DB(self.H, self.G, self.K[1],
                                              E=self.E, bias=self.bias)
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0:  # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.G, dimReadout[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.nonlinearityReadout())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                    bias=self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.


        self.n_classes = n_classes
        if self.n_classes is not None:
            assert n_nodes is not None # we need to know how big this fc will be
            # We add a final global fully connected that maps down stuff from "outputDim" to "n_classes"
            self.global_fc = nn.Linear(dimOutputSignals*n_nodes, n_classes)

    def splitForward(self, x, S):

        # Check the dimensions of the input
        #   S: B x T (x E) x N x N
        #   x: B x T x F[0] x N
        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        B = S.shape[0]
        T = S.shape[1]
        assert S.shape[2] == self.E
        N = S.shape[3]
        assert S.shape[4] == N

        assert len(x.shape) == 4
        assert x.shape[0] == B
        assert x.shape[1] == T
        assert x.shape[2] == self.F
        assert x.shape[3] == N

        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        z0 = torch.randn((B, self.H, N), device=x.device)

        # Add the GSO for each graph filter
        self.hiddenState.addGSO(S)
        self.outputState.addGSO(S)

        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(x, z0)
        # Compute the output trajectory from the hidden states
        yOut = self.outputState(z)
        yOut = self.rho(yOut)  # Don't forget the nonlinearity!
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2)  # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y)  # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N

    def forward(self, x, S):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x, S)

        if self.n_classes is not None:
            # Custom code Gabriele Mazzola
            # We will map the output values as a whole down to the number of classes for classification,
            # and we apply cross-entropy to obtain a probability distribution

            # The output is B x T x (dimFeatures[-1] or dimReadout[-1]) x N
            # we're only interested in the last output (B x (dimFeatures[-1] or dimReadout[-1]) x N)
            last_output = output[:, -1, :, :]
            last_output_flat = last_output.reshape(last_output.shape[0], -1)

            global_output = self.global_fc(last_output_flat)

            assert global_output.shape[1] == self.n_classes
            return global_output

        return output

    def singleNodeForward(self, x, S, nodes):

        # x is of shape B x T x F[0] x N
        batchSize = x.shape[0]
        N = x.shape[3]

        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
               or type(nodes) is list \
               or type(nodes) is np.ndarray

        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x 1 x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x T x dimRedout[-1] x 1
        # and we just squeeze the last dimension

        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select

        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype=np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                 for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize

        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, 1, N, 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype=x.dtype,
                                       device=x.device)

        # Now compute the output
        y = self.forward(x, S)
        # This output is of size B x T x dimReadout[-1] x N

        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1

        # Squeeze the last dimension and return
        return y.squeeze(3)