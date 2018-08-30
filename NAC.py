import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module

class NAC(Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = Parameter(torch.Tensor(n_out, n_in))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)
    
    def forward(self, input):
        weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, weights)