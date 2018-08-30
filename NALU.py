import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module
from NAC import NAC

class NALU(Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.NAC = NAC(n_in, n_out)
        self.G = Parameter(torch.Tensor(1, n_in))
        self.eps = 1e-6
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.G)
    
    def forward(self, input):
        g = torch.sigmoid(F.linear(input, self.G))
        y1 = g * self.NAC(input)
        y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self.eps)))
        return y1 + y2