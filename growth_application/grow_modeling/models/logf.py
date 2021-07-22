from . utils import safe_exp


class Logf:

    """
    Logistic function (Logf)
    """
    x0 = 0, 1, 1

    @staticmethod
    def forward(t, param):
        t0, h1, k = safe_exp(param)
        pred = h1 / (1 + safe_exp(-k*(t-t0)))
        return pred
