import numpy as np


class Linf:

    bounds = [(-np.inf, np.inf) for _ in range(2)]
    x0 = 0, 1

    @staticmethod
    def forward(t, param):
        beta0, beta1 = param
        pred = beta0 + beta1 * t
        return pred




