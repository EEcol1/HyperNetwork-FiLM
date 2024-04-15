import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import random
from utils import element_Linear,Linear2
# ----------------------------------------------------------------------------
# MAIN NETWORK
def set_random_seed(random_seed: int):
    seed=random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

class MainNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[400, 400], use_bias=True, use_hnet=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_bias = use_bias
        self.use_hnet= use_hnet

        # Parameter shapes
        self._param_shapes = []
        self._hypershapes_learned = []
        

        layers=[]
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            in_features = dims[i]
            out_features = dims[i + 1]
            self._param_shapes += [[out_features, in_features]]
            if use_bias:
                self._param_shapes += [[out_features]]
            if use_hnet is False:
                layers += [nn.Linear(in_features, out_features, bias=use_bias)]
                if i < len(dims) - 2:
                    layers += [nn.ReLU()]
        
        self.layers=nn.Sequential(*layers)


    @property
    def num_params(self):
        if len(self._param_shapes) == 0:
            return 0
        return np.sum([np.prod(l) for l in self._param_shapes])
    
   
            

    def forward(self, x, weights=[]):
        assert (isinstance(weights, list) and len(
            weights) == len(self._param_shapes)) or self.use_hnet is False
        assert all([list(w.shape) == self._param_shapes[i]
                   for i, w in enumerate(weights)]) or self.use_hnet is False

        if self.use_hnet is False:
            return self.layers(x)


        w_list = []
        b_list = []
        for params in weights:
            if params.ndim == 2:
                w_list += [params]
            elif params.ndim == 1:
                assert self.use_bias
                b_list += [params]
               
        
               
        num_layers = len(w_list)
        for layer_idx in range(num_layers):
            weight = w_list[layer_idx]
            bias = None
            if self.use_bias:
                bias = b_list[layer_idx]
            x = F.linear(x, weight=weight, bias=bias)
            
            if layer_idx < num_layers - 1:
                x=F.relu(x)

        return x
# ----------------------------------------------------------------------------
# HYPER NETWORK
#input:x(1028,), task_idx(0-9)
#ouput: weight in a reshaped form.
class HyperNetwork(nn.Module):
    def __init__(self, target_shapes, num_tasks, task_emb_dim, hidden_dims=[100, 100], use_bias=True):
        super().__init__()
        self.target_shapes = target_shapes
        self.num_tasks = num_tasks
        self.task_emb_dim = task_emb_dim
        self.hidden_dims = hidden_dims
        self.use_bias = use_bias
        
        # Task Embeddings
        self.input_emb=nn.Parameter(torch.randn(task_emb_dim))

        num_target_params = np.sum([np.prod(l) for l in self.target_shapes])
        
        
        # Layers
        input_dim = self.task_emb_dim
        output_dim = num_target_params
        dims = [input_dim] + hidden_dims + [output_dim]
        
        self.linear_layers=nn.ModuleList()
        self.element_linear_layers=nn.ModuleList()
        for i in range(len(dims)-1):
            in_features=dims[i]
            out_features=dims[i+1]
            self.linear_layers.append(nn.Linear(in_features, out_features, bias=self.use_bias))
            if i <len(dims)-2:
                self.element_linear_layers.append(element_Linear(out_features,num_tasks))
                #self.element_linear_layers.append(Linear2(out_features,out_features,num_tasks))
                

        self.linear_parameters=[]
        for linear_layer in self.linear_layers:
            self.linear_parameters+=linear_layer.parameters()
        
        self.element_parameters=[]
        for element_layer in self.element_linear_layers:
            self.element_parameters+=element_layer.parameters()
        
        
    
    @property
    def num_params(self):
        return np.sum([np.prod(p.size()) for p in self.parameters()])
        
    # WARNING: We assume that entire batch "x" belongs to the same task "task_idx"
    def forward(self, task_id,dTheta=None):
        assert task_id in range(self.num_tasks)
        # HyperNetwork Embeddings
        if isinstance(task_id,int):
            task_id = task_id* torch.ones(1, dtype=torch.long, device="cuda") #task_idx multipy???
            
        task_emb = self.input_emb.view(1,-1)                  # [1, task_emb_dim]
        if dTheta is not None:
            assert(len(dTheta)==len(self.theta))
            with torch.no_grad():
                for i,p in enumerate(self.layers.parameters()):
                    p.data+=dTheta[i]
                

                

        x=task_emb
        for i, linear_layer in enumerate(self.linear_layers):
            x=linear_layer(x)
            if i<len(self.linear_layers)-2:
                x=nn.ReLU()(x)
            if task_id>0 and i<len(self.element_linear_layers):
                element_linear_layer=self.element_linear_layers[i]
                x=element_linear_layer(x,task_id)
        mnet_params=x.view(-1)
        
        # Reshape parameters
        current_idx = 0
        out = []
        for shape in self.target_shapes:
            num_params = int(np.prod(shape))
            out += [mnet_params[current_idx:current_idx + num_params].view(*shape)]
            current_idx += num_params
        
        return out
        
