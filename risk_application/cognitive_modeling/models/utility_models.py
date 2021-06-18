import numpy as np


def u_pow(x, theta): return x**theta


def u_exp(x, theta): return 1 - np.exp(-theta*x)


def u_lin(x, theta=None): return x
