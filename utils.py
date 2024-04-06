
import torch
import numpy as np
import math
import torch.nn as nn
assert torch.cuda.is_available(), "CUDA device not detected!"

CW5_sequence=[
    "hammer", 
    "push-wall", 
    "faucet-close", 
    "handle-press-side", 
    "window-close"
]

class element_Linear(nn.Module):
    def __init__(self,in_features,nb_tasks,bias: bool =True, device=None,dtype=None)->None:
        super().__init__()
        self.in_features=in_features
        self.weight=nn.Parameter(torch.randn((nb_tasks,in_features)))
        if bias==True:
            self.bias=nn.Parameter(torch.randn((nb_tasks,in_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x,task_id):
        if task_id==0:
            return x
        return x*self.weight[task_id].reshape(1,-1)+self.bias[task_id].reshape(1,-1)