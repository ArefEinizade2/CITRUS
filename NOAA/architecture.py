import numpy as np
import torch
import torch.nn as nn

from utils import GGCRNNCell
from graph_utils import permutation_by_degree

zeroTolerance = 1e-9  # Values below this number are considered zero.


class GatedGCRNNforClassification(nn.Module):
    """
    GatedGCRNNfforClassification: implements the full GCRNN architecture, i.e.
    h_t = sigma(\hat{Q_t}(A(S)*x_t) + \check{Q_t}(B(S)*h_{t-1}))
    y_t = rho(C(S)*h_t)
    where:
     - h_t, x_t, y_t are the state, input and output variables respectively
     - sigma and rho are the state and output nonlinearities
     - \hat{Q_t} and \check{Q_t} are the input and forget gate operators, which could be time, node or edge gates (or
     time+node, time+edge)
     - A(S), B(S) and C(S) are the input-to-state, state-to-state and state-to-output graph filters
     In the classification version of the Gated GCRNN, y_t is a C-dimensional one-hot vector, where C is the number of classes

    Initialization:

        GatedGCRNNforClassification(inFeatures, stateFeatures, inputFilterTaps,
             stateFilterTaps, stateNonlinearity,
             outputNonlinearity,
             dimLayersMLP,
             GSO,
             bias,
             time_gating=True,
             spatial_gating=None,
             mlpType = 'oneMlp'
             finalNonlinearity = None,
             dimNodeSignals=None, nFilterTaps=None,
             nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None)

        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            inputFilterTaps (int): number of input filter taps
            stateFilterTaps (int): number of state filter taps
            stateNonlinearity (torch.nn): sigma, state nonlinearity in the GRNN cell
            outputNonlinearity (torch.nn): rho, module from torch.nn nonlinear activations
            dimLayersMLP (list of int): number of hidden units of the MLP layers
            GSO (np.array): graph shift operator
            bias (bool): include bias after graph filter on every layer
            time_gating (bool): time gating flag, default True
            spatial_gating (string): None, 'node' or 'edge'
            mlpType (string): either 'oneMlp' or 'multipMLP'; 'multipMLP' corresponds to local MLPs, one per node
            finalNonlinearity (torch.nn): nonlinearity to apply to y, if any (e.g. softmax for classification)
            dimNodeSignals (list of int): dimension of the signals at each layer of C(S) if it is a GNN
            nFilterTaps (list of int): number of filter taps on each layer of C(S) if it is a GNN
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer of the C(S) if it is a Selection GNN
            poolingFunction (nn.Module in Utils.graphML): summarizing function of C(S) if it is a GNN with pooling
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer if C(S) is a GNN with pooling
            maxN (int): maximum number of neighborhood exchanges if C(S) is an Aggregation GNN (default: None)

        Output:
            nn.Module with a full GRNN architecture, state + output neural networks,
            with the above specified characteristics.

    Forward call:

        GatedGCRNNforClassification(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data of shape
                batchSize x seqLength x dimFeatures x numberNodes
    """

    def __init__(self,
                 # State GCRNN
                 inFeatures, stateFeatures, inputFilterTaps,
                 stateFilterTaps, stateNonlinearity,
                 # Output NN nonlinearity
                 outputNonlinearity,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Gating
                 time_gating=True,
                 spatial_gating=None,
                 # Final nonlinearity, if any, to apply to y
                 finalNonlinearity=None,
                 # Output NN filtering if output NN is GNN
                 dimNodeSignals=None, nFilterTaps=None,
                 # Output NN pooling is output NN is GNN with pooling
                 nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN=None):

        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2
        assert GSO.shape[0] == GSO.shape[1]
        GSO, order = permutation_by_degree(GSO)
        self.order = order
        GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]])  # 1 x N x N


        # Store the values for the state GRNN (using the notation in the paper):
        self.F_i = inFeatures  # Input features
        self.K_i = inputFilterTaps  # Filter taps of input filter
        self.F_h = stateFeatures  # State features
        self.K_h = stateFilterTaps  # Filter taps of state filter
        self.E = GSO.shape[0]  # Number of edge features
        self.N = GSO.shape[1]  # Number of nodes
        self.bias = bias  # Boolean
        self.time_gating = time_gating  # Boolean
        self.spatial_gating = spatial_gating  # None, "node" or "edge"
        self.S = torch.tensor(GSO)
        self.sigma1 = stateNonlinearity
        # Declare State GCRNN
        self.stateGCRNN = GGCRNNCell(self.F_i, self.F_h, self.K_i,
                                         self.K_h, self.sigma1, self.time_gating, self.spatial_gating,
                                         self.E, self.bias)
        self.stateGCRNN.addGSO(self.S)
        # Dimensions of output GNN's  lfully connected layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output neural network nonlinearity
        self.sigma2 = outputNonlinearity
        # Selection/Aggregation GNN parameters for the output neural network (default None)
        self.F_o = dimNodeSignals
        self.K_o = nFilterTaps
        self.nSelectedNodes = nSelectedNodes
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.maxN = maxN
        # Nonlinearity to apply to the output, e.g. softmax for classification (default None)
        self.sigma3 = finalNonlinearity

        # \\ If output neural network is MLP:
        if dimNodeSignals is None and nFilterTaps is None:
            fc = []
            if len(self.dimLayersMLP) > 0:  # Maybe we don't want to MLP anything
                # The first layer has to connect whatever was left of the graph
                # signal, flattened.
                dimInputMLP = self.N * self.F_h
                # (i.e., N nodes, each one described by F_h features,
                # which means this will be flattened into a vector of size
                # N x F_h)
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias=self.bias))
                for l in range(len(dimLayersMLP) - 1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l + 1],
                                        bias=self.bias))
                    # And we're done
            # Declare output MLP
            if self.sigma3 is not None:
                fc.append(self.sigma3())
            self.outputNN = nn.Sequential(*fc)
            # so we finally have the architecture.

        else:
            raise Exception("We only allow for MLP in the final part of the network")

    def forward(self, x, h0):
        # Now we compute the forward call
        # x is of dimensions: [batchSize x seqLength x dimFeatures x numberNodes]
        x = x[:, :, :, self.order] # let's reorder


        H = self.stateGCRNN(x, h0)
        h = H.select(1, -1)
        if self.F_o is None:  # outputNN is MLP
            h = h.view(-1, self.F_h * self.N)
            assert h.shape[0] == x.shape[0]
            y = self.outputNN(h)
        else:
            y = self.outputNN(h)
        return y

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)


class GatedGCRNNforRegression(nn.Module):
    """
    GatedGCRNNforRegression: implements the full GCRNN architecture, i.e.
    h_t = sigma(\hat{Q_t}(A(S)*x_t) + \check{Q_t}(B(S)*h_{t-1}))
    y_t = rho(C(S)*h_t)
    where:
     - h_t, x_t, y_t are the state, input and output variables respectively
     - sigma and rho are the state and output nonlinearities
     - \hat{Q_t} and \check{Q_t} are the input and forget gate operators, which could be time, node or edge gates (or
     time+node, time+edge)
     - A(S), B(S) and C(S) are the input-to-state, state-to-state and state-to-output graph filters
     In the regression version of the Gated GCRNN, y_t is a graph signal
    Initialization:
        GatedGCRNNforRegression(inFeatures, stateFeatures, inputFilterTaps,
             stateFilterTaps, stateNonlinearity,
             outputNonlinearity,
             dimLayersMLP,
             GSO,
             bias,
             time_gating=True,
             spatial_gating=None,
             mlpType = 'oneMlp'
             finalNonlinearity = None,
             dimNodeSignals=None, nFilterTaps=None,
             nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None)
        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            inputFilterTaps (int): number of input filter taps
            stateFilterTaps (int): number of state filter taps
            stateNonlinearity (torch.nn): sigma, state nonlinearity in the GRNN cell
            outputNonlinearity (torch.nn): rho, module from torch.nn nonlinear activations
            dimLayersMLP (list of int): number of hidden units of the MLP layers
            GSO (np.array): graph shift operator
            bias (bool): include bias after graph filter on every layer
            time_gating (bool): time gating flag, default True
            spatial_gating (string): None, 'node' or 'edge'
            mlpType (string): either 'oneMlp' or 'multipMLP'; 'multipMLP' corresponds to local MLPs, one per node
            finalNonlinearity (torch.nn): nonlinearity to apply to y, if any (e.g. softmax for classification)
            dimNodeSignals (list of int): dimension of the signals at each layer of C(S) if it is a GNN
            nFilterTaps (list of int): number of filter taps on each layer of C(S) if it is a GNN
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer of the C(S) if it is a Selection GNN
            poolingFunction (nn.Module in Utils.graphML): summarizing function of C(S) if it is a GNN with pooling
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer if C(S) is a GNN with pooling
            maxN (int): maximum number of neighborhood exchanges if C(S) is an Aggregation GNN (default: None)
        Output:
            nn.Module with a full GRNN architecture, state + output neural networks,
            with the above specified characteristics.
    Forward call:
        GatedGCRNNforRegression(x)
        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes
        Output:
            y (torch.tensor): output data of shape
                batchSize x seqLength x dimFeatures x numberNodes
    """

    def __init__(self,
                 # State GCRNN
                 inFeatures, stateFeatures, inputFilterTaps,
                 stateFilterTaps, stateNonlinearity,
                 # Output NN nonlinearity
                 outputNonlinearity,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Gating
                 time_gating=True,
                 spatial_gating=None,
                 # Type of MLP, one for all nodes or one, local, per node
                 mlpType='oneMlp',
                 # Final nonlinearity, if any, to apply to y
                 finalNonlinearity=None,
                 # Output NN filtering if output NN is GNN
                 dimNodeSignals=None, nFilterTaps=None,
                 # Output NN pooling is output NN is GNN with pooling
                 nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN=None):

        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2
        assert GSO.shape[0] == GSO.shape[1]
        GSO, order = permutation_by_degree(GSO)
        self.order = order
        GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]])  # 1 x N x N


        # Store the values for the state GRNN (using the notation in the paper):
        self.F_i = inFeatures  # Input features
        self.K_i = inputFilterTaps  # Filter taps of input filter
        self.F_h = stateFeatures  # State features
        self.K_h = stateFilterTaps  # Filter taps of state filter
        self.E = GSO.shape[0]  # Number of edge features
        self.N = GSO.shape[1]  # Number of nodes
        self.bias = bias  # Boolean
        self.time_gating = time_gating  # Boolean
        self.spatial_gating = spatial_gating  # None, "node" or "edge"
        self.S = torch.tensor(GSO)
        self.sigma1 = stateNonlinearity
        # Declare State GCRNN
        self.stateGCRNN = GGCRNNCell(self.F_i, self.F_h, self.K_i,
                                         self.K_h, self.sigma1, self.time_gating, self.spatial_gating,
                                         self.E, self.bias)
        self.stateGCRNN.addGSO(self.S)
        # Dimensions of output GNN's  lfully connected layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output neural network nonlinearity
        self.sigma2 = outputNonlinearity
        # Selection/Aggregation GNN parameters for the output neural network (default None)
        self.F_o = dimNodeSignals
        self.K_o = nFilterTaps
        self.nSelectedNodes = nSelectedNodes
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.maxN = maxN
        # Nonlinearity to apply to the output, e.g. softmax for classification (default None)
        self.sigma3 = finalNonlinearity
        # Type of MLP
        self.mlpType = mlpType

        # \\ If output neural network is MLP:
        if dimNodeSignals is None and nFilterTaps is None:
            fc = []
            if len(self.dimLayersMLP) > 0:  # Maybe we don't want to MLP anything
                if mlpType == 'oneMlp':
                    # The first layer has to connect whatever was left of the graph
                    # signal, flattened.
                    dimInputMLP = self.N * self.F_h
                    # (i.e., N nodes, each one described by F_h features,
                    # which means this will be flattened into a vector of size
                    # N x F_h)
                elif mlpType == 'multipMlp':
                    # one perceptron per node, same parameters across all of them
                    dimInputMLP = self.F_h
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias=self.bias))
                for l in range(len(dimLayersMLP) - 1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l + 1],
                                        bias=self.bias))
                    # And we're done
            # Declare output MLP
            if self.sigma3 is not None:
                fc.append(self.sigma3())
            self.outputNN = nn.Sequential(*fc)
            # so we finally have the architecture.

        else:
            raise Exception("Shouldn't go here")

    def forward(self, x, h0):
        # Now we compute the forward call
        # x is of dimensions: [batchSize x seqLength x dimFeatures x numberNodes]
        x = x[:, :, :, self.order]  # let's reorder

        batchSize = x.shape[0]
        seqLength = x.shape[1]


        H = self.stateGCRNN(x, h0)
        # flatten by merging batch and sequence length dimensions
        flatH = H.view(-1, self.F_h, self.N)

        if self.F_o is None:  # outputNN is MLP
            if self.mlpType == 'multipMlp':
                flatH = flatH.view(-1, self.F_h, self.N)
                flatH = flatH.transpose(1, 2)
                flatY = torch.empty(0)
                for i in range(self.N):
                    hNode = flatH.narrow(1, i, 1)
                    hNode = hNode.squeeze()
                    yNode = self.outputNN(hNode)
                    yNode = yNode.unsqueeze(1)
                    flatY = torch.cat([flatY, yNode], 1)
                flatY = flatY.transpose(1, 2)
                flatY = flatY.squeeze()
            elif self.mlpType == 'oneMlp':
                flatH = flatH.view(-1, self.F_h * self.N)
                flatY = self.outputNN(flatH)
        else:
            flatY = self.outputNN(flatH)
        # recover original batch and sequence length dimensions
        y = flatY.view(batchSize, seqLength, -1)
        return y[:, -1, :]

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
