import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   
    
from Needed_Functions.early_stop_solver import EarlyStopInt
from Needed_Functions.block_constant import ConstantODEblock
from Needed_Functions.function_laplacian_diffusion import LaplacianODEFunc
from torch_geometric.nn import GCNConv

import diffusion_net
from diffusion_net.utils import toNP
from diffusion_net.geometry import to_basis, from_basis
from torch_geometric.nn import GCNConv
import networkx as nx
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.contiguous().view(combined_shape)
#%%
class GCN_diff(torch.nn.Module):
    def __init__(self, use_gdc, in_channels,hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels, cached=True, normalize=not use_gdc)
    
    def forward(self,x,edge_index,edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()

        return x
#%% 

class Time_derivative_diffusion(nn.Module):
    def __init__(self, n_block, GCN_opt, k, C_inout, num_nodes, single_t, method='spectral'):
        super(Time_derivative_diffusion, self).__init__()
        self.C_inout = C_inout
        self.k = k
        self.single_t = single_t
        
        # same t for all channels
        if self.single_t:
            self.diffusion_time = nn.Parameter(torch.Tensor(1))
        else:
        # learnable t
            self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  

        
        self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        self.method = method # one of ['spectral', 'implicit_dense']
        self.num_nodes = num_nodes 
        
        nn.init.constant_(self.diffusion_time, 0.0)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        self.Conv_layer.reset_parameters()            
        nn.init.constant_(self.diffusion_time, 0.0)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
        
        
        
    def forward(self, x, edge_index, L, mass, evals, evecs):
         

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':
              
            # Transform to spectral
            x_spec = torch.matmul(torch.transpose(evecs,1,0),x)
            
            # Diffuse
            time = self.diffusion_time
            
            # Same t for all channels
            if self.single_t:
                dim = x.shape[1]
                time = time.repeat(dim)

            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            
            x_diffuse_spec = diffusion_coefs * x_spec

            x_diffuse_spec = x_diffuse_spec -(self.alpha)*x_spec # Skip connection
            x_diffuse = torch.matmul(evecs, x_diffuse_spec)
            x_diffuse = x_diffuse + (self.betta)*x
           
            x_diffuse = self.Conv_layer(x_diffuse, L._indices(), edge_weight=L._values()).relu()  

                      
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            L=torch.tensor(L.todense())
            mat_dense = L.unsqueeze(1).expand(self.C_inout, V, V).clone()
            
            # mat_dense = L.to_dense().expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse

#%%            

class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )

#%%

    
class TIDE_block(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, n_block, k, C_in, C_width, C_out, mlp_hidden_dims, num_nodes,
                 dropout=True,
                 diffusion_method='spectral',
                 single_t = False,
                 use_gdc = [],
                 with_MLP=True,
                 device='cpu'):
        super(TIDE_block, self).__init__()

        # Specified dimensions
        self.k = k
        self.C_width = C_width
        self.C_in = C_in
        self.C_out = C_out
        self.mlp_hidden_dims = mlp_hidden_dims
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.dropout = dropout
        self.with_MLP = with_MLP
        self.num_nodes = num_nodes
        self.n_block = n_block
        self.device = device
        self.MLP_C = 2*self.C_width

        self.diff_derivative = Time_derivative_diffusion(self.n_block, self.use_gdc, self.k, self.C_width, self.num_nodes, self.single_t, method=diffusion_method)      
        

        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)
      
    def forward(self, epoch, x_in, x_original, edge_index, mass, L, evals, evecs, x0):

        # Manage dimensions
        if x_in.shape[-1] != self.C_width:
            raise ValueError("Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(x_in.shape, self.C_width))

        x_diff_derivative = self.diff_derivative(x_in, edge_index, L, mass, evals, evecs)   

        x_diffuse= x_diff_derivative
            

        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        if self.with_MLP:
            x0_out = self.mlp(feature_combined)
            x0_out = x0_out + x_in 
        else:
            x0_out = x_diffuse + x_in 

        return x0_out      

#%%
class TIDE_net(nn.Module):

    def __init__(self,k, C_in, C_out, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu'):   
        super(TIDE_net, self).__init__()

        # Basic parameters
        self.k = k
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.num_nodes = num_nodes
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.device = device
   

        # Outputs
        self.last_activation = last_activation
        
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        
        #MLP
        self.with_MLP = with_MLP
        
       
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
            
        # TIDE blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = TIDE_block(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, epoch, x_in, edge_index, mass, L=None, evals=None, evecs=None, edges=None, faces=None):
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(epoch, x, x_in, edge_index, mass, L, evals, evecs, x_in)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        return x_out  
#%%
class Time_derivative_diffusion_product(nn.Module):
    def __init__(self, n_block, GCN_opt, k, C_inout, num_nodes, single_t, method='spectral'):
        super(Time_derivative_diffusion_product, self).__init__()
        self.P = len(num_nodes) # number of factor graphs
        self.C_inout = C_inout
        self.k = k
        self.single_t = single_t
        
        # same t for all channels
        if self.single_t:
            self.diffusion_time = nn.Parameter(torch.Tensor(self.P))
        else:
        # learnable t
            self.diffusion_time = nn.Parameter(torch.Tensor(self.P, self.C_inout))

        
        self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        self.method = method # one of ['spectral', 'implicit_dense']
        self.num_nodes = num_nodes 
        
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        self.Conv_layer.reset_parameters()            
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
        
        
        
    def forward(self, x, edge_index, edge_weight, mass, evals, evecs):
         

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # print(x.shape)
            # print(evecs.shape)
                        
            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            if self.single_t:
                # Diffuse
                # time = self.diffusion_time
                # evals_kroned = torch.exp(-self.diffusion_time[0] * evals[0])
                # for p  in range(1, self.P):
                #     evals_kroned = torch.kron(evals_kroned, torch.exp(-self.diffusion_time[p] * evals[p]))
                # diffusion_coefs = evals_kroned.repeat(x.shape[-1], 1).T
                
                evals_kroned_cos = torch.cos(-self.diffusion_time[0] * evals[0])
                for p  in range(1, self.P):
                    evals_kroned_cos = torch.kron(evals_kroned_cos, torch.cos(-self.diffusion_time[p] * evals[p]))

                evals_kroned_sin = torch.cos(-self.diffusion_time[0] * evals[0])
                for p  in range(1, self.P):
                    evals_kroned_sin = torch.kron(evals_kroned_sin, torch.cos(-self.diffusion_time[p] * evals[p]))

                evals_kroned = evals_kroned_cos - evals_kroned_sin

                diffusion_coefs = evals_kroned.repeat(x.shape[-1], 1).T
            else:                 
                diffusion_coefs = torch.zeros(x_spec.shape[1:]).to(device)
                for c in range(self.C_inout):
                    # evals_kroned = torch.exp(-self.diffusion_time[0, c] * evals[0])
                    # for p  in range(1, self.P):
                    #     evals_kroned = torch.kron(evals_kroned, torch.exp(-self.diffusion_time[p, c] * evals[p]))
                    # diffusion_coefs[:, c] = evals_kroned

                    evals_kroned_cos = torch.cos(-self.diffusion_time[0, c] * evals[0])
                    for p  in range(1, self.P):
                        evals_kroned_cos = torch.kron(evals_kroned_cos, torch.cos(-self.diffusion_time[p, c] * evals[p]))
                        
                    evals_kroned_sin = torch.cos(-self.diffusion_time[0, c] * evals[0])
                    for p  in range(1, self.P):
                        evals_kroned_sin = torch.kron(evals_kroned_sin, torch.sin(-self.diffusion_time[p, c] * evals[p]))
                        
                    evals_kroned = evals_kroned_cos - evals_kroned_sin
                        
                    diffusion_coefs[:, c] = evals_kroned
            

            # x_diffuse = self.Conv_layer(x_diffuse, edge_index, edge_weight=edge_weight).relu()  
            x_diffuse_spec = diffusion_coefs * x_spec


            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec.to(torch.float32), evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]
            L = edge_weight
            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse

#%%
class CPGNN_block(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, n_block, k, C_in, C_width, C_out, mlp_hidden_dims, num_nodes,
                 dropout=True,
                 diffusion_method='spectral',
                 single_t = False,
                 use_gdc = [],
                 with_MLP=True,
                 device='cpu'):
        super(CPGNN_block, self).__init__()
        
        # Specified dimensions
        self.k = k
        self.C_width = C_width
        self.C_in = C_in
        self.C_out = C_out
        self.mlp_hidden_dims = mlp_hidden_dims
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.dropout = dropout
        self.with_MLP = with_MLP
        self.num_nodes = num_nodes
        self.n_block = n_block
        self.device = device
        self.MLP_C = self.C_width        
        self.channel_mixer = nn.Linear(self.C_width, self.C_width)

        self.diff_derivative = Time_derivative_diffusion_product(self.n_block, self.use_gdc, self.k, 
                                                                 self.C_width, self.num_nodes, self.single_t,
                                                                 method=diffusion_method)      

        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)

    def forward(self, epoch, x_in, x_original, edge_index, mass, L, evals, evecs, x0):

        # Manage dimensions
        if x_in.shape[-1] != self.C_width:
            raise ValueError("Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(x_in.shape, self.C_width))

        x_diff_derivative = self.diff_derivative(x_in, edge_index, L, mass, evals, evecs)   

        x_diffuse = x_diff_derivative
            

        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)
            x0_out = self.mlp(feature_combined)
            x0_out = x0_out + x_in 
        else:
            x0_out = x_diffuse + x_in 

        return x0_out      
#%%
class CPGNN_block_v2(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, n_block, k, C_in, C_width, C_out, mlp_hidden_dims, num_nodes,
                 dropout=True,
                 diffusion_method='spectral',
                 single_t = False,
                 use_gdc = [],
                 with_MLP=True,
                 device='cpu'):
        super(CPGNN_block_v2, self).__init__()
        
        # Specified dimensions
        self.k = k
        self.C_width = C_width
        self.C_in = C_in
        self.C_out = C_out
        self.mlp_hidden_dims = mlp_hidden_dims
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.dropout = dropout
        self.with_MLP = with_MLP
        self.num_nodes = num_nodes
        self.n_block = n_block
        self.device = device
        self.MLP_C = self.C_width        
        self.channel_mixer = nn.Linear(self.C_width, self.C_width)

        self.diff_derivative = Time_derivative_diffusion_product(self.n_block, self.use_gdc, self.k, 
                                                                 self.C_width, self.num_nodes, self.single_t,
                                                                 method=diffusion_method)      

        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([2*self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)

    def forward(self, epoch, x_in, x_original, edge_index, mass, edge_weight, evals, evecs, x0):

        # Manage dimensions
        if x_in.shape[-1] != self.C_width:
            raise ValueError("Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(x_in.shape, self.C_width))

        x_diff_derivative = self.diff_derivative(x_in, edge_index, edge_weight, mass, evals, evecs)   

        x_diffuse = x_diff_derivative
            

        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)
            x0_out = self.mlp(feature_combined)
            x0_out = x0_out + x_in 
        else:
            x0_out = x_diffuse + x_in 

        return x0_out      

#%%
class CPGNN(nn.Module):

    def __init__(self, k, C_in, C_out, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CPGNN, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
   
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)


        # CPGNN blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = CPGNN_block(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, epoch, x_org, edge_index, mass, L=None, evals=None, evecs=None, edges=None, faces=None):
        # Apply the first linear layer
        x = self.first_lin(x_org)

        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(epoch, x, x_org, edge_index, mass, L, evals, evecs, x_org)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1)))
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        return x_out  
#%%
class CPGNN_ST(nn.Module):

    def __init__(self, k, C_in, C_out, edge_index, edge_weight, mass, evals, evecs, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CPGNN_ST, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mass = mass
        self.evals = evals
        self.evecs = evecs
        
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)
        self.MergeParams = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1]))


        self.node_embeddings = NodeEmbedding(self.num_nodes[0], 4)
        
        # CPGNN blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = CPGNN_block_v2(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, x, L=None):
        # Apply the first linear layer
        # print(x.shape)
        Batch, M_AR, Nodes, Feat = x.shape
        # Concatenate node embeddings to input
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        # print(x.shape)
        x_org = x
        # x = self.first_lin(torch.cat([x_org, emb], -1))
        x = self.first_lin(x_org) # --> b t*n c_width

        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

        
        # --> b t*n c_width
        
        
        # Apply the last linear layer        
        x_out = self.last_lin(x) # --> b t*n h
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1))) # --> b num_classes
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out) # --> b t*n h

        # print(x_out.shape)
        x_out = x_out.contiguous().view((Batch, M_AR, Nodes, self.C_out)) # --> b t n h
        x_out = torch.transpose(x_out, 3, 1) # --> b h n t
        # print(x_out.shape)
        x_out = torch.matmul(x_out, self.MergeParams) # --> b h n
        # print(x_out.shape)
        # x_out = torch.mean(x_out, 1)
        # print(x_out.shape)
        x_out = torch.unsqueeze(x_out, -1) # --> b h n 1
        # print(x_out.shape)
        return x_out  
#%%
class CPGNN_ST_v2(nn.Module):

    def __init__(self, k, C_in, C_out, edge_index, edge_weight, mass, evals, evecs, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CPGNN_ST_v2, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mass = mass
        self.evals = evals
        self.evecs = evecs
        
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.second_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, 1)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)
        self.MergeM = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1], self.C_out))
        self.MergeHorizons = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1]))


        self.node_embeddings = NodeEmbedding(self.num_nodes[0], 4)

        # CPGNN blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = CPGNN_block_v2(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, x, L=None):
        # Apply the first linear layer
        # print(x.shape)
        Batch, M_AR, Nodes, Feat = x.shape
        x_tensor = x
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        x = magic_combine(x_tensor, 1, 3) # combine dimension 1, 2
        # print(x.shape)
        x_org = x
        x = self.first_lin(x_org)
        # x = self.first_lin(torch.cat([x_org, emb], -1))
        # print(x.shape)
        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1)))
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # print(x_out.shape)
        x_out = x_out.contiguous().view((Batch, M_AR, Nodes, 1))
        x_out = torch.transpose(x_out, 3, 1)
        # print(x_out.shape)
        x_out = torch.matmul(x_out, self.MergeM[:, 0])
        # print(x_out.shape)
        # x_out = torch.mean(x_out, 1)
        # print(x_out.shape)
        x_out = torch.unsqueeze(x_out, -1)
        # print(x_out.shape)
        x_tensor = torch.concatenate((x_tensor[:,1:,:,:], x_out), 1)
        x_out_H = x_out
        
        
        for h in range(1, self.C_out):
            x = magic_combine(x_tensor, 1, 3) # combine dimension 1, 2
            # print(x.shape)
            x_org = x
            # print(x_org.shape)
            x = self.first_lin(x_org)

            
            # Apply each of the blocks
            for b in self.blocks:
                # x = self.med_linear(x)
                x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

            # Apply the last linear layer        
            x_out = self.last_lin(x)
            
            if self.graph_wise:
                # x_out = torch.mean(x_out, 1)
                x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1)))
            
            # Apply last nonlinearity if specified
            if self.last_activation != None:
                x_out = self.last_activation(x_out)

            # print(x_out.shape)
            x_out = x_out.contiguous().view((Batch, M_AR, Nodes, 1))
            x_out = torch.transpose(x_out, 3, 1)
            # print(x_out.shape)
            x_out = torch.matmul(x_out, self.MergeM[:, h])
            # print(x_out.shape)
            # x_out = torch.mean(x_out, 1)
            # print(x_out.shape)
            x_out = torch.unsqueeze(x_out, -1)
            # print(x_out.shape)
            x_tensor = torch.concatenate((x_tensor[:,1:,:,:], x_out), 1) 
            x_out_H = torch.concatenate((x_out_H, x_out), 1)
        return x_out_H  

#%%
class CPGNN_ST_v3(nn.Module):

    def __init__(self, k, C_in, C_out, edge_index, edge_weight, mass, evals, evecs, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CPGNN_ST_v3, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mass = mass
        self.evals = evals
        self.evecs = evecs
        
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, 1)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)
        self.MergeM = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1], self.C_out))
        self.MergeHorizons = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1]))


        # CPGNN blocks
        self.blocks = []
        for h in range(C_out):
            self.blocks_h = []
            for i_block in range(self.N_block):
                block = CPGNN_block_v2(n_block = i_block+1+h,
                                                k = k,
                                                C_in= C_in,
                                                C_width = C_width, 
                                                C_out=C_out,
                                                mlp_hidden_dims = mlp_hidden_dims,
                                                num_nodes = num_nodes,
                                                dropout = dropout,
                                                diffusion_method = diffusion_method,
                                                single_t = single_t,
                                                use_gdc = use_gdc,
                                                with_MLP = with_MLP,
                                                device = self.device)
    
                self.blocks_h.append(block)
                self.add_module("block_"+str(i_block)+"_h_"+str(h), self.blocks_h[-1])
            self.blocks.append(self.blocks_h)

    def forward(self, x, L=None):
        # Apply the first linear layer
        # print(x.shape)
        Batch, M_AR, Nodes, Feat = x.shape
        x_tensor = x
        x = magic_combine(x_tensor, 1, 3) # combine dimension 1, 2
        # print(x.shape)
        x_org = x
        x = self.first_lin(x_org)

        blocks_h = self.blocks[0]
        # Apply each of the blocks
        for b in blocks_h:
            # x = self.med_linear(x)
            x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1)))
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # print(x_out.shape)
        x_out = x_out.contiguous().view((Batch, M_AR, Nodes, 1))
        x_out = torch.transpose(x_out, 3, 1)
        # print(x_out.shape)
        x_out = torch.matmul(x_out, self.MergeM[:, 0])
        # print(x_out.shape)
        # x_out = torch.mean(x_out, 1)
        # print(x_out.shape)
        x_out = torch.unsqueeze(x_out, -1)
        # print(x_out.shape)
        x_tensor = torch.concatenate((x_tensor[:,1:,:,:], x_out), 1)
        x_out_H = x_out
        
        
        for h in range(1, self.C_out):
            blocks_h = self.blocks[h]
            x = magic_combine(x_tensor, 1, 3) # combine dimension 1, 2
            # print(x.shape)
            x_org = x
            x = self.first_lin(x_org)

            
            # Apply each of the blocks
            for b in blocks_h:
                # x = self.med_linear(x)
                x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

            # Apply the last linear layer        
            x_out = self.last_lin(x)
            
            if self.graph_wise:
                # x_out = torch.mean(x_out, 1)
                x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1)))
            
            # Apply last nonlinearity if specified
            if self.last_activation != None:
                x_out = self.last_activation(x_out)

            # print(x_out.shape)
            x_out = x_out.contiguous().view((Batch, M_AR, Nodes, 1))
            x_out = torch.transpose(x_out, 3, 1)
            # print(x_out.shape)
            x_out = torch.matmul(x_out, self.MergeM[:, h])
            # print(x_out.shape)
            # x_out = torch.mean(x_out, 1)
            # print(x_out.shape)
            x_out = torch.unsqueeze(x_out, -1)
            # print(x_out.shape)
            x_tensor = torch.concatenate((x_tensor[:,1:,:,:], x_out), 1) 
            x_out_H = torch.concatenate((x_out_H, x_out), 1)
        return x_out_H 
#%%
class CPGNN_ST_in_TTS(nn.Module):

    def __init__(self, k, C_in, C_out, edge_index, edge_weight, mass, evals, evecs, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CPGNN_ST_in_TTS, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mass = mass
        self.evals = evals
        self.evecs = evecs
        
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)
        self.MergeParams = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1]))


        self.node_embeddings = NodeEmbedding(self.num_nodes[0], 4)
        
        # CPGNN blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = CPGNN_block_v2(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, x, L=None):
        # Apply the first linear layer
        # print(x.shape)
        Batch, M_AR, Nodes, Feat = x.shape
        # Concatenate node embeddings to input
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        # print(x.shape)
        x_org = x
        # x = self.first_lin(torch.cat([x_org, emb], -1))
        x = self.first_lin(x_org) # --> b t*n c_width

        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

        
        # --> b t*n c_width
        
        
        # Apply the last linear layer        
        x_out = self.last_lin(x) # --> b t*n h
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1))) # --> b num_classes
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out) # --> b t*n h

        # print(x_out.shape)
        x_out = x_out.contiguous().view((Batch, M_AR, Nodes, self.C_out)) # --> b t n h
        x_out = torch.transpose(x_out, 3, 1) # --> b h n t
        # print(x_out.shape)
        x_out = torch.matmul(x_out, self.MergeParams) # --> b h n
        x_out = torch.transpose(x_out, 2, 1) # --> b n f
        return x_out  

#%%
class SoCITRUS(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 emb_size: int = 16,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2,
                 edge_index=None,
                 edge_weight=None,
                 mass = None,
                 evals = None,
                 evecs = None,
                 C_width = 4,
                 N_block = 1,
                 single_t = True,
                 use_gdc = [],
                 num_nodes = [],
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):
        super(SoCITRUS, self).__init__()

        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)

        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)

        # STMP
        # self.time_nn = RNN(input_size=hidden_size,
        #                    hidden_size=hidden_size,
        #                    n_layers=rnn_layers,
        #                    cell='gru',
        #                    return_only_last_state=True)

        # self.space_nn = DiffConv(in_channels=hidden_size,
        #                          out_channels=hidden_size,
        #                          k=gnn_kernel,
        #                          root_weight=True)
        #CPGNN:
        self.CPGNN = CPGNN_ST_in_TTS(None, hidden_size, hidden_size, edge_index, edge_weight, 
                     mass, evals, evecs, C_width, N_block, single_t, use_gdc, num_nodes, 
                     last_activation, mlp_hidden_dims, dropout, with_MLP, 
                     diffusion_method, device, graph_wise)
        
        # Decoder
        self.decoder = nn.Linear(hidden_size + emb_size, input_size * horizon)

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # print(x.shape)
        # x: [batch time nodes features]
        b, t, n, f = x.size()
        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)
        # Encoder
        x_enc = self.encoder(x_emb)  # linear proj: x_enc = [x||emb]Θ + b  --> b t n f
        # STMP
        # h = self.time_nn(x_enc)  # temporal processing: x=[b t n f] -> h=[b n f]
        # z = self.space_nn(h, edge_index, edge_weight)  # spatial processing --> b n f
        # CPGNN
        z = self.CPGNN(x_enc) # --> b n f
        # Decoder
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1) # concatenate node embeddings to z
        x_out = self.decoder(z_emb)  # linear proj: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        # print(x_horizon.shape)
        return x_horizon
#%%
class GCN(torch.nn.Module):
    def __init__(self, Fea, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.Fea = Fea
        self.conv1 = GCNConv(self.Fea[0], self.Fea[1])
        self.convs = []
        for i in range(1, self.num_layers + 1):
            self.convs.append(GCNConv(self.Fea[i], self.Fea[i+1]).to(device))
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x)
        x = F.tanh(x)
        for i in range(1, self.num_layers + 1):
            x = self.convs[i-1](x, edge_index)
            # x = Conv
            x = F.dropout(x)
            x = F.tanh(x)
        return x
#%%
import torch
from einops.layers.torch import Rearrange
from torch import nn
# from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import MLP, ResidualMLP
from tsl.nn.functional import expand_then_cat
from tsl.nn.utils import get_layer_activation
# from tsl.utils.parser_utils import ArgParser, str_to_bool

# from lib.sgp_preprocessing import sgp_spatial_embedding


class SGPModel(nn.Module):
    def __init__(self,
                 input_size,
                 order,
                 n_nodes,
                 hidden_size,
                 mlp_size,
                 output_size,
                 n_layers,
                 horizon,
                 positional_encoding,
                 emb_size=32,
                 exog_size=None,
                 resnet=False,
                 fully_connected=False,
                 dropout=0.,
                 activation='silu'):
        super(SGPModel, self).__init__()

        if fully_connected:
            out_channels = hidden_size
            self.input_encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )
        else:
            out_channels = hidden_size - hidden_size % order
            self.input_encoder = nn.Sequential(
                # [b n f] -> [b 1 n f]
                Rearrange('b n f -> b f n '),
                nn.Conv1d(in_channels=input_size,
                          out_channels=out_channels,
                          kernel_size=1,
                          groups=order),
                Rearrange('b f n -> b n f'),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )

        if resnet:
            self.mlp = ResidualMLP(
                input_size=out_channels,
                hidden_size=mlp_size,
                exog_size=exog_size,
                n_layers=n_layers,
                activation=activation,
                dropout=dropout,
                parametrized_skip=True
            )
        else:
            self.mlp = MLP(
                input_size=out_channels,
                n_layers=n_layers,
                hidden_size=mlp_size,
                exog_size=exog_size,
                activation=activation,
                dropout=dropout
            )

        if positional_encoding:
            self.node_emb = StaticGraphEmbedding(
                n_tokens=n_nodes,
                emb_size=emb_size
            )
            self.lin_emb = nn.Linear(emb_size, out_channels)

        else:
            self.register_parameter('node_emb', None)
            self.register_parameter('lin_emb', None)

        self.readout = LinearReadout(
            input_size=mlp_size,
            output_size=output_size,
            horizon=horizon,
        )

    def forward(self, x, u=None, node_index=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = x[:, -1] if x.ndim == 4 else x
        x = self.input_encoder(x)
        if self.node_emb is not None:
            x = x + self.lin_emb(self.node_emb(token_index=node_index))
        if u is not None:
            u = u[:, -1] if u.ndim == 4 else u
            x = expand_then_cat([x, u], dim=-1)
        x = self.mlp(x)

        return self.readout(x)

    # @staticmethod
    # def add_model_specific_args(parser: ArgParser):
    #     parser.opt_list('--hidden-size', type=int, default=32, tunable=True,
    #                     options=[16, 32, 64, 128, 256])
    #     parser.opt_list('--mlp-size', type=int, default=32, tunable=True,
    #                     options=[16, 32, 64, 128, 256])
    #     parser.opt_list('--emb-size', type=int, default=32, tunable=True,
    #                     options=[16, 32, 64])
    #     parser.opt_list('--n-layers', type=int, default=1, tunable=True,
    #                     options=[1, 2, 3])
    #     parser.opt_list('--dropout', type=float, default=0., tunable=True,
    #                     options=[0., 0.2, 0.3])
    #     parser.opt_list('--fully-connected', type=str_to_bool, nargs='?',
    #                     const=True, default=False)
    #     parser.opt_list('--positional-encoding', type=str_to_bool, nargs='?',
    #                     const=True, default=False)
    #     parser.opt_list('--resnet', type=str_to_bool, nargs='?', const=True,
    #                     default=False)
    #     return parser
#%%
#%%
class GTCNN_block(torch.nn.Module):
    def __init__(self, K, F_in, F_out, num_layers, Adj_t, Adj_s):
        super(GTCNN_block, self).__init__()
        self.K = K
        self.num_layers = num_layers
        self.F_in = F_in
        self.F_out = F_out
        self.lin_first = nn.Linear(self.F_in, self.F_out)
        self.lin_last = nn.Linear(self.F_out, self.F_out)
        self.H = nn.Parameter(torch.Tensor(self.num_layers, self.K, self.F_out, self.F_out))    
        self.Adj_t = Adj_t
        self.Adj_s = Adj_s
        self.s = nn.Parameter(torch.Tensor(4)).to('cuda')
        self.I_t = torch.eye(self.Adj_t.shape[0]).to('cuda')
        self.I_s = torch.eye(self.Adj_s.shape[0]).to('cuda')
        self.I_st = torch.eye(self.Adj_t.shape[0]*self.Adj_s.shape[0]).to('cuda')
        self.MergeParams = torch.nn.parameter.Parameter(torch.randn(self.Adj_t.shape[0]))
        
    def forward(self, x):
        
        
        Batch, M_AR, Nodes, Feat = x.shape
        Adj = self.s[0]*self.I_st + self.s[1]*torch.kron(self.I_t, self.Adj_s) + self.s[2]*torch.kron(self.Adj_t, self.I_s) + self.s[3]*torch.kron(self.Adj_t, self.Adj_s)
        x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        x = self.lin_first(x) # --> b t*n f_out
        for l in range(self.num_layers):
            Sum = 0
            for k in range(self.K):
                H_kl = torch.squeeze(self.H[l, k, :, :])
                Sum = Sum + torch.matmul(torch.matmul(torch.matrix_power(Adj, k), x), H_kl)
            x = torch.tanh(Sum) # --> b t*n f
        # H_kl = torch.squeeze(self.H[0, 0, :, :])
        # x = torch.matmul(torch.matmul(Adj, x), H_kl)
        x = self.lin_last(x) # --> b t*n f_out
        x_out = x.contiguous().view((Batch, M_AR, Nodes, self.F_out)) # --> b t n h
        x_out = torch.transpose(x_out, 3, 1) # --> b h n t
        # print(x_out.shape)
        x_out = torch.matmul(x_out, self.MergeParams) # --> b h n
        x_out = torch.transpose(x_out, 2, 1) # --> b n f
        return x_out
#%%
class GTCNN_TTS(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 emb_size: int = 16,
                 hidden_size: int = 32, Adj_t = None, Adj_s = None, K = 3,
                 N_block = 1, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu'):
        super(GTCNN_TTS, self).__init__()

        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)

        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)

        #CPGNN:
        self.GTCNN = GTCNN_block(K=K, F_in=hidden_size, F_out=hidden_size, num_layers=N_block,
                                 Adj_t=Adj_t, Adj_s=Adj_s)
        
        # Decoder
        self.decoder = nn.Linear(hidden_size + emb_size, input_size * horizon)

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x):
        # print(x.shape)
        # x: [batch time nodes features]
        b, t, n, f = x.size()
        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)
        # Encoder
        x_enc = self.encoder(x_emb)  # linear proj: x_enc = [x||emb]Θ + b  --> b t n f
        # STMP
        # h = self.time_nn(x_enc)  # temporal processing: x=[b t n f] -> h=[b n f]
        # z = self.space_nn(h, edge_index, edge_weight)  # spatial processing --> b n f
        # CPGNN
        z = self.GTCNN(x_enc) # --> b n f
        # Decoder
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1) # concatenate node embeddings to z
        x_out = self.decoder(z_emb)  # linear proj: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        # print(x_horizon.shape)
        return x_horizon
#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class Time_derivative_diffusion_CGNN(nn.Module):
    def __init__(self, n_block, GCN_opt, k, C_inout, num_nodes, single_t, method='spectral'):
        super(Time_derivative_diffusion_CGNN, self).__init__()
        self.P = len(num_nodes) # number of factor graphs
        self.C_inout = C_inout
        self.k = k
        self.single_t = single_t
        
        # same t for all channels
        if self.single_t:
            self.diffusion_time = nn.Parameter(torch.Tensor(1))
        else:
        # learnable t
            self.diffusion_time = nn.Parameter(torch.Tensor(self.C_inout))

        
        self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        self.method = method # one of ['spectral', 'implicit_dense']
        self.num_nodes = num_nodes 
        
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        self.Conv_layer.reset_parameters()            
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
        
        
        
    def forward(self, x, edge_index, edge_weight, mass, evals, evecs):
         

        # if x.shape[-1] != self.C_inout:
        #     raise ValueError(
        #         "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
        #             x.shape, self.C_inout))

        if self.method == 'spectral':

            # print(x.shape)
            # print(evecs.shape)
                        
            # Transform to spectral
            # x_spec = to_basis(x, evecs, mass)
            
            # x_diffuse_spec = diffusion_coefs * x_spec

            # Transform to spectral
            x_spec = torch.matmul(torch.transpose(evecs,1,0),x)
            
            # Diffuse
            time = self.diffusion_time
            
            # Same t for all channels
            if self.single_t:
                dim = x.shape[-1]
                time = time.repeat(dim)

            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            
            # print(diffusion_coefs.shape)
            # print(x_spec.shape)
            x_diffuse_spec = diffusion_coefs * x_spec

            # x_diffuse_spec = x_diffuse_spec -(self.alpha)*x_spec # Skip connection
            x_diffuse = torch.matmul(evecs, x_diffuse_spec)
            # x_diffuse = x_diffuse + (self.betta)*x
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]
            L = edge_weight
            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse
#%%
class CGNN_block_v2(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, n_block, k, C_in, C_width, C_out, mlp_hidden_dims, num_nodes,
                 dropout=True,
                 diffusion_method='spectral',
                 single_t = False,
                 use_gdc = [],
                 with_MLP=True,
                 device='cpu'):
        super(CGNN_block_v2, self).__init__()
        
        # Specified dimensions
        self.k = k
        self.C_width = C_width
        self.C_in = C_in
        self.C_out = C_out
        self.mlp_hidden_dims = mlp_hidden_dims
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.dropout = dropout
        self.with_MLP = with_MLP
        self.num_nodes = num_nodes
        self.n_block = n_block
        self.device = device
        self.MLP_C = self.C_width        
        self.channel_mixer = nn.Linear(self.C_width, self.C_width)

        self.diff_derivative = Time_derivative_diffusion_CGNN(self.n_block, self.use_gdc, self.k, 
                                                                 self.C_width, self.num_nodes, self.single_t,
                                                                 method=diffusion_method)      

        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([2*self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)

    def forward(self, epoch, x_in, x_original, edge_index, mass, edge_weight, evals, evecs, x0):

        # Manage dimensions
        # if x_in.shape[-1] != self.C_width:
        #     raise ValueError("Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(x_in.shape, self.C_width))

        x_diff_derivative = self.diff_derivative(x_in, edge_index, edge_weight, mass, evals, evecs)   

        x_diffuse = x_diff_derivative
            

        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)
            # print(self.C_width)
            # print(feature_combined.shape)
            x0_out = self.mlp(feature_combined)
            x0_out = x0_out + x_in 
        else:
            x0_out = x_diffuse + x_in 

        return x0_out      

#%%
class CGNN_ST_in_TTS(nn.Module):

    def __init__(self, k, C_in, C_out, edge_index, edge_weight, mass, evals, evecs, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CGNN_ST_in_TTS, self).__init__()

        # Basic parameters
        self.k = k # number of selected eigen-vectors/values in EVD
        self.C_in = C_in # number of input channels
        self.C_out = C_out # number of out channels (e.g.num_nodes_list, number of classes)
        self.C_width = C_width # number of channels for the output of each block or layer
        self.N_block = N_block # number of blocks
        self.num_nodes = num_nodes # number of nodes
        self.single_t = single_t # t in heat kernels
        self.use_gdc = use_gdc #???
        self.device = device # device (e.g., CPU, GPU, etc)
        self.graph_wise = graph_wise
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mass = mass
        self.evals = evals
        self.evecs = evecs
        
        # Outputs
        self.last_activation = last_activation
    
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")


        #MLP
        self.with_MLP = with_MLP

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
        self.merge_lin = nn.Linear(torch.prod(torch.tensor(self.num_nodes)), 1)
        self.MergeParams = torch.nn.parameter.Parameter(torch.randn(self.num_nodes[1]))


        self.node_embeddings = NodeEmbedding(self.num_nodes[0], 4)
        
        # CPGNN blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = CGNN_block_v2(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, x, L=None):
        # Apply the first linear layer
        # print(x.shape)
        # x = x.squeeze()
        # if len(x.shape==4):
        #     Batch, M_AR, Nodes, Fea = x.shape
        # else:
        #     Batch, Nodes, Fea = x.shape
                
        # x = torch.transpose(x, 2, 1)
        # print(x.shape)
        # Concatenate node embeddings to input
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        # x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        # print(x.shape)
        x_org = x
        # x = self.first_lin(torch.cat([x_org, emb], -1))
        x = self.first_lin(x_org) # --> b t*n c_width

        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(0, x, x_org, self.edge_index, self.mass, self.edge_weight, self.evals, self.evecs, x_org)

        
        # --> b t*n c_width
        
        
        # Apply the last linear layer        
        x_out = self.last_lin(x) # --> b t*n h
        
        if self.graph_wise:
            # x_out = torch.mean(x_out, 1)
            x_out = torch.squeeze(self.merge_lin(x_out.transpose(2, 1))) # --> b num_classes
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out) # --> b t*n h

        # print(x_out.shape)
        # x_out = x_out.contiguous().view((Batch, M_AR, Nodes, self.C_out)) # --> b t n h
        # x_out = torch.transpose(x_out, 3, 1) # --> b h n t
        # print(x_out.shape)
        # x_out = torch.matmul(x_out, self.MergeParams) # --> b h n
        # x_out = torch.transpose(x_out, 2, 1) # --> b n f
        return x_out  

#%%
class CGNN_STT(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 emb_size: int = 16,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2,
                 edge_index=None,
                 edge_weight=None,
                 mass = None,
                 evals = None,
                 evecs = None,
                 C_width = 4,
                 N_block = 1,
                 single_t = True,
                 use_gdc = [],
                 num_nodes = [],
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):
        super(CGNN_STT, self).__init__()

        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)

        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)

        # STMP
        self.time_nn = RNN(input_size=hidden_size,
                            hidden_size=hidden_size,
                            n_layers=rnn_layers,
                            cell='gru',
                            return_only_last_state=True)

        # self.space_nn = DiffConv(in_channels=hidden_size,
        #                          out_channels=hidden_size,
        #                          k=gnn_kernel,
        #                          root_weight=True)
        #CPGNN:
        self.CGNN = CGNN_ST_in_TTS(None, hidden_size, hidden_size, edge_index, edge_weight, 
                     mass, evals, evecs, C_width, N_block, single_t, use_gdc, num_nodes, 
                     last_activation, mlp_hidden_dims, dropout, with_MLP, 
                     diffusion_method, device, graph_wise)
        
        # Decoder
        self.decoder = nn.Linear(hidden_size + emb_size, input_size * horizon)

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # print(x.shape)
        # x: [batch time nodes features]
        b, t, n, f = x.size()
        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)
        # Encoder
        x_enc = self.encoder(x_emb)  # linear proj: x_enc = [x||emb]Θ + b  --> b t n f
        # STMP
        # h = self.time_nn(x_enc)  # temporal processing: x=[b t n f] -> h=[b n f]
        # z = self.space_nn(h, edge_index, edge_weight)  # spatial processing --> b n f
        # CPGNN
        # print('z.shape: ', z.shape)
        h = self.CGNN(x_enc) # --> b n f
        # print(h.shape)
        z = self.time_nn(h)  # temporal processing: x=[b t n f] -> h=[b n f]
        # print(z.shape)
        # Decoder
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1) # concatenate node embeddings to z
        x_out = self.decoder(z_emb)  # linear proj: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        # print(x_horizon.shape)
        return x_horizon
#%%
class CGNN_TTS(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 emb_size: int = 16,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2,
                 edge_index=None,
                 edge_weight=None,
                 mass = None,
                 evals = None,
                 evecs = None,
                 C_width = 4,
                 N_block = 1,
                 single_t = True,
                 use_gdc = [],
                 num_nodes = [],
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):
        super(CGNN_TTS, self).__init__()

        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)

        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)

        # STMP
        self.time_nn = RNN(input_size=hidden_size,
                            hidden_size=hidden_size,
                            n_layers=rnn_layers,
                            cell='gru',
                            return_only_last_state=True)

        # self.space_nn = DiffConv(in_channels=hidden_size,
        #                          out_channels=hidden_size,
        #                          k=gnn_kernel,
        #                          root_weight=True)
        #CPGNN:
        self.CGNN = CGNN_ST_in_TTS(None, hidden_size, hidden_size, edge_index, edge_weight, 
                     mass, evals, evecs, C_width, N_block, single_t, use_gdc, num_nodes, 
                     last_activation, mlp_hidden_dims, dropout, with_MLP, 
                     diffusion_method, device, graph_wise)
        
        # Decoder
        self.decoder = nn.Linear(hidden_size + emb_size, input_size * horizon)

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # print(x.shape)
        # x: [batch time nodes features]
        b, t, n, f = x.size()
        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)
        # Encoder
        x_enc = self.encoder(x_emb)  # linear proj: x_enc = [x||emb]Θ + b  --> b t n f
        # STMP
        # h = self.time_nn(x_enc)  # temporal processing: x=[b t n f] -> h=[b n f]
        # z = self.space_nn(h, edge_index, edge_weight)  # spatial processing --> b n f
        # CPGNN
        # print('z.shape: ', z.shape)
        # print(h.shape)
        h = self.time_nn(x_enc)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.CGNN(h) # --> b n f
        # print(z.shape)
        # Decoder
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1) # concatenate node embeddings to z
        x_out = self.decoder(z_emb)  # linear proj: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        # print(x_horizon.shape)
        return x_horizon