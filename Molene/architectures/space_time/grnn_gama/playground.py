import numpy as np

import torch

from architectures.space_time.grnn_gama.grnn import GraphRecurrentNN_DB

torch.cuda.current_device()
import torch; torch.set_default_dtype(torch.float64)

n_nodes = 58
n_batches = 50
n_timesteps = 25
gso = np.zeros((n_batches, n_timesteps, 1, n_nodes, n_nodes))
gso[:, :, :] = np.identity(n_nodes)
gso = torch.from_numpy(gso)


input_features = 3
input_values = torch.randn(size=(n_batches, n_timesteps, input_features, n_nodes))


n_classes = 45

grnn_model = GraphRecurrentNN_DB(
# Graph filtering
                 dimInputSignals = input_features,
                 dimOutputSignals = 8,
                 dimHiddenSignals = 20,
                 nFilterTaps = [4, 4], bias = True,
                 # Nonlinearities
                 nonlinearityHidden = torch.tanh,
                 nonlinearityOutput = torch.tanh,
                 nonlinearityReadout = torch.nn.ReLU(),  # nn.Module
                 # Local MLP in the end
                 dimReadout = [],
                 # Structure
                 dimEdgeFeatures=1,
    n_classes=n_classes,
    n_nodes=gso.shape[-1]
) # Structure

print(grnn_model)





y = grnn_model(input_values, gso)
print(y.shape)


#         Input:
#             x (torch.tensor): input data of shape
#                 batchSize x timeSamples x dimInputSignals x numberNodes
#             GSO (torch.tensor): graph shift operator; shape
#                 batchSize x timeSamples (x dimEdgeFeatures)
#                                                     x numberNodes x numberNodes
#         Output:
#             y (torch.tensor): output data after being processed by the GRNN;
#                 batchSize x timeSamples x dimReadout[-1] x numberNodes


print(sum(p.numel() for p in grnn_model.parameters() if p.requires_grad))