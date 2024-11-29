import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   
    
from early_stop_solver import EarlyStopInt
from block_constant import ConstantODEblock
from function_laplacian_diffusion import LaplacianODEFunc
from torch_geometric.nn import GCNConv

import diffusion_net
from diffusion_net.utils import toNP
from diffusion_net.geometry import to_basis, from_basis
from torch_geometric.nn import GCNConv

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
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

        
        # self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        self.method = method # one of ['spectral', 'implicit_dense']
        self.num_nodes = num_nodes 
        
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        # self.Conv_layer.reset_parameters()            
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform(self.diffusion_time)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
        
        
        
    def forward(self, x, edge_index, L, mass, evals, evecs):
         

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            if self.single_t:
                # Diffuse
                # time = self.diffusion_time
                evals_kroned = torch.exp(-self.diffusion_time[0] * evals[0])
                for p  in range(1, self.P):
                    evals_kroned = torch.kron(evals_kroned, torch.exp(-self.diffusion_time[p] * evals[p]))
                diffusion_coefs = evals_kroned.repeat(x.shape[-1], 1).T
            else:                 
                diffusion_coefs = torch.zeros(x_spec.shape).to(device)
                for c in range(self.C_inout):
                    evals_kroned = torch.exp(-self.diffusion_time[0, c] * evals[0])
                    for p  in range(1, self.P):
                        evals_kroned = torch.kron(evals_kroned, torch.exp(-self.diffusion_time[p, c] * evals[p]))
                    diffusion_coefs[:, c] = evals_kroned

            
            # diffusion_coefs = torch.exp(-evals_kroned)
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec.to(torch.float32), evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

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
            feature_combined = x_diffuse
        
        # Apply the mlp
        if self.with_MLP:
            # x0_out = self.mlp(feature_combined)
            x0_out = torch.tanh(self.channel_mixer(x_diffuse)) 
            
            # x0_out = x0_out + x_original 
        else:
            x0_out = x_diffuse

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
class CITRUS(nn.Module):

    def __init__(self, k, C_in, C_out, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu', graph_wise=False):   
        super(CITRUS, self).__init__()

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
            x_out = torch.mean(x_out, 1)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        return x_out  
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
        self.last_lin = nn.Linear(Fea[-1], Fea[-1])
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x)
        x = F.tanh(x)
        for i in range(1, self.num_layers + 1):
            x = self.convs[i-1](x, edge_index)
            # x = Conv
            x = F.dropout(x)
        x = self.last_lin(x)
        return x
    #%%
class GTCNN(torch.nn.Module):
    def __init__(self, K, F_in, hidden_f, F_out, num_layers, Adj_t, Adj_s):
        super(GTCNN, self).__init__()
        self.K = K
        self.num_layers = num_layers
        self.F_in = F_in
        self.F_out = F_out
        self.hidden_f = hidden_f
        self.lin_first = nn.Linear(self.F_in, self.hidden_f).to('cuda')
        self.lin_last = nn.Linear(self.hidden_f, self.F_out).to('cuda')
        self.H = nn.Parameter(torch.Tensor(self.num_layers, self.K, self.hidden_f, self.hidden_f)).to('cuda')    
        # self.H = nn.Parameter(torch.Tensor(self.num_layers, 2, 3)).to('cuda')    
        degrees = np.sum(Adj_t, axis=1)
        inverse_sqrt_degree_matrix = np.linalg.inv(np.sqrt(np.diag(degrees)))
        Adj_t = np.dot(np.dot(inverse_sqrt_degree_matrix, Adj_t), inverse_sqrt_degree_matrix)/2
        self.Adj_t = Adj_t.to('cuda')
        degrees = np.sum(Adj_s, axis=1)
        inverse_sqrt_degree_matrix = np.linalg.inv(np.sqrt(np.diag(degrees)))
        Adj_s = np.dot(np.dot(inverse_sqrt_degree_matrix, Adj_s), inverse_sqrt_degree_matrix)/2
        self.Adj_s = Adj_s.to('cuda')
        self.s = nn.Parameter(torch.Tensor(4)).to('cuda')
        self.I_t = torch.eye(self.Adj_t.shape[0]).to('cuda')
        self.I_s = torch.eye(self.Adj_s.shape[0]).to('cuda')
        self.I_st = torch.eye(self.Adj_t.shape[0]*self.Adj_s.shape[0]).to('cuda')
        self.Adj = self.s[0]*self.I_st + self.s[1]*torch.kron(self.I_t, self.Adj_s) + self.s[2]*torch.kron(self.Adj_t, self.I_s) + self.s[3]*torch.kron(self.Adj_t, self.Adj_s)
        # self.Adj = torch.kron(self.I_t, self.Adj_s) + self.s[2]*torch.kron(self.Adj_t, self.I_s)
        self.Adj = self.Adj.to('cuda')
        
    def forward(self, x):
        
        
        # M_AR_times_Nodes, Feat = x.shape # --> b t*n f
        # Concatenate node embeddings to input
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        # x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        x = self.lin_first(x) # --> b t*n f_out
        for l in range(self.num_layers):
            Sum = torch.tensor(0.0).float()
            for k in range(self.K):
                H_kl = torch.squeeze(self.H[l, k, :, :])
                # Adj = self.I_st.float()
                # for i in range(k-1):
                #     Adj = torch.matmul(Adj, self.Adj.float())
                # Sum = Sum + torch.matmul(torch.matmul(Adj, x.float()), H_kl.float())
                Sum = Sum + torch.matmul(torch.matmul(self.Adj.float(), x.float()), H_kl.float())
            x = torch.tanh(Sum) # --> b t*n f_out

        # for l in range(self.num_layers):
        #     Sum = torch.tensor(0.0).float()
        #     for k in range(2):
        #         for k2 in range(2):
        #             H_kl = torch.squeeze(self.H[l, k, k2])
        #             # Adj = self.I_st.float()
        #             # for i in range(k-1):
        #             #     Adj = torch.matmul(Adj, self.Adj.float())
        #             # Sum = Sum + torch.matmul(torch.matmul(Adj, x.float()), H_kl.float())
        #             Sum = Sum + H_kl*torch.matmul(torch.kron(torch.matrix_power(self.Adj_t.float(), k), torch.matrix_power(self.Adj_s.float(), k2)), x.float())
        #     x = torch.tanh(Sum) # --> b t*n f_out

        x = self.lin_last(x) # --> b t*n f_out
        return x
#%%
class Time_derivative_diffusion_CGNN(nn.Module):
    def __init__(self, n_block, GCN_opt, k, C_inout, num_nodes, single_t, method='spectral'):
        super(Time_derivative_diffusion_CGNN, self).__init__()
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
           
            # x_diffuse = self.Conv_layer(x_diffuse, L._indices(), edge_weight=L._values()).relu()  

                      
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
class CGNN_block(nn.Module):
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
        super(CGNN_block, self).__init__()

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

        self.diff_derivative = Time_derivative_diffusion_CGNN(self.n_block, self.use_gdc, self.k, self.C_width, self.num_nodes, self.single_t, method=diffusion_method)      
        

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
class CGNN(nn.Module):

    def __init__(self,k, C_in, C_out, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu'):   
        super(CGNN, self).__init__()

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
            block = CGNN_block(n_block = i_block+1,
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
class GTCNN_block(torch.nn.Module):
    def __init__(self, K, C_in, C_width, Adj_t, Adj_s, with_MLP=True):
        super(GTCNN_block, self).__init__()
        self.K = K
        self.C_in = C_in
        self.C_width = C_width
        self.with_MLP = with_MLP
        # self.lin_first = nn.Linear(self.F_in, self.hidden_f).to('cuda')
        # self.lin_last = nn.Linear(self.hidden_f, self.F_out).to('cuda')
        self.H = nn.Parameter(torch.Tensor(self.K, self.C_width, self.C_width)).to('cuda')    
        nn.init.normal_(self.H)
        # self.H = nn.Parameter(torch.Tensor(self.num_layers, 2, 3)).to('cuda')    
        self.Adj_t = Adj_t.to('cuda')
        self.Adj_s = Adj_s.to('cuda')
        self.s = nn.Parameter(torch.Tensor(4)).to('cuda')
        nn.init.normal_(self.s)
        self.I_t = torch.eye(self.Adj_t.shape[0]).to('cuda')
        self.I_s = torch.eye(self.Adj_s.shape[0]).to('cuda')
        self.I_st = torch.eye(self.Adj_t.shape[0]*self.Adj_s.shape[0]).to('cuda')
        self.Adj = self.s[0]*self.I_st + self.s[1]*torch.kron(self.I_t, self.Adj_s) + self.s[2]*torch.kron(self.Adj_t, self.I_s) + self.s[3]*torch.kron(self.Adj_t, self.Adj_s)
        # self.Adj = torch.kron(self.I_t, self.Adj_s) + self.s[2]*torch.kron(self.Adj_t, self.I_s)
        self.Adj = self.Adj.to('cuda')
        self.MLP_C = self.C_in + self.C_width
        self.mlp_hidden_dims = [self.C_width]
        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=None)
        
            
    def reset_parameters(self):
        nn.init.normal(self.H)
        nn.init.normal(self.s)

    def forward(self, x, x_in):
        
        # M_AR_times_Nodes, Feat = x.shape # --> b t*n f
        # Concatenate node embeddings to input
        # emb = self.node_embeddings(expand=(Batch, M_AR, -1, -1))
        # emb = magic_combine(emb, 1, 3)
        # x = magic_combine(x, 1, 3) # combine dimension 1, 2 --> b t*n f
        # x = self.lin_first(x) # --> b t*n f_out
        Sum = torch.tensor(0.0).float()
        for k in range(self.K):
            H_kl = torch.squeeze(self.H[k, :, :])
            # Adj = self.I_st.float()
            # for i in range(k-1):
            #     Adj = torch.matmul(Adj, self.Adj.float())
            # Sum = Sum + torch.matmul(torch.matmul(Adj, x.float()), H_kl.float())
            Sum = Sum + torch.matmul(torch.matmul(torch.matrix_power(self.Adj.float(), k), x.float()), H_kl.float())
        x = torch.tanh(Sum) # --> b t*n f_out'
        
        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x, x_in), dim=-1)
            x0_out = self.mlp(feature_combined)
            # x0_out = x0_out + x_in 
        else:
            x0_out = x
        return x0_out
#%%
class GTCNN_v2(nn.Module):

    def __init__(self,K, C_in, C_out, C_width, N_block, Adj_t, Adj_s, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu'):   
        super(GTCNN_v2, self).__init__()

        # Basic parameters
        self.K = K
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.num_nodes = num_nodes
        self.use_gdc = use_gdc
        self.device = device
        degrees = torch.sum(Adj_t, axis=1)
        inverse_sqrt_degree_matrix = torch.linalg.inv(np.sqrt(torch.diag(degrees)))
        Adj_t = inverse_sqrt_degree_matrix @ Adj_t @ inverse_sqrt_degree_matrix/2
        self.Adj_t = Adj_t.to('cuda')
        degrees = torch.sum(Adj_s, axis=1)
        inverse_sqrt_degree_matrix = torch.linalg.inv(np.sqrt(torch.diag(degrees)))
        Adj_s = inverse_sqrt_degree_matrix @ Adj_s @ inverse_sqrt_degree_matrix/2
        self.Adj_s = Adj_s.to('cuda')

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
            block = GTCNN_block(self.K, self.C_in, self.C_width, self.Adj_t, self.Adj_s, with_MLP = self.with_MLP)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in):
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(x, x_in)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        return x_out  