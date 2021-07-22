import numpy as np
from . utils import safe_exp

class Jpps:

    """
    Jolicoeur, Pontier, Pernin and Sempé (Jpps)

    P. Jolicoeur, J. Pontier, M.-O. Pernin, and M. Sempé. A lifetime asymptotic
    growth curve for human height. Biometrics, pages 995–1003, 1988.
    https://doi.org/10.2307/2531730
    """

    x0 = 0, 0, 0, 0, 0, 0, 0

    @staticmethod
    def forward(t, param):

        h1, C1, C2, C3, D1, D2, D3 = np.log(1 + safe_exp(param))

        tp = t + 0.75

        pred = h1 * (1 - 1 / (
                    1 + (tp / D1) ** C1 + (tp / D2) ** C2 + (tp / D3) ** C3))
        return pred
