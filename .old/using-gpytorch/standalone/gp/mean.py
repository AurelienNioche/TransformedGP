import torch


class ZeroMean:
    def __call__(self, x):
        return torch.zeros(x.shape[0])
