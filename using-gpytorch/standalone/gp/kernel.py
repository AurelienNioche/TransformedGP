import torch
from torch import nn


class ScaledRBFKernel(nn.Module):

    def __init__(self, jitter=1e-03):
        super(ScaledRBFKernel, self).__init__()

        self.raw_outputscale = torch.nn.Parameter(torch.zeros(1))
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(1))
        self.jitter = jitter

        self.softplus = torch.nn.Softplus()

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        outputscale = self.softplus(self.raw_outputscale)
        lengthscale = self.softplus(self.raw_lengthscale)
        return outputscale * self.kernel(x1, x2, lengthscale) \
            + self.jitter * torch.eye(x1.size(0))

    @staticmethod
    def kernel(x1, x2, lengthscale):
        delta = torch.pow((x1[:, None] - x2) / lengthscale, 2.0)
        return torch.exp(-0.5 * delta)
