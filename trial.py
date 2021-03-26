# import os
# import time

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy

# import jax
# from jax import vmap
# import jax.numpy as jnp
# import jax.random as random
#
# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS, init_to_feasible, init_to_median, \
#     init_to_sample, \
#     init_to_uniform, init_to_value
#
# from pymc3.gp.util import plot_gp_dist
import pymc3 as pm
import pymc3.sampling_jax

import numpyro
numpyro.set_platform("cpu")

# from tqdm import tqdm

# Don't sting my eyes
sns.set()


# Default to double precision.
# numpyro.enable_x64()

# Utility models

def u_pow(x, theta): return x ** theta


def u_exp(x, theta): return 1 - np.exp(-theta * x)


def u_lin(x, theta=None): return x


# Generate choice data

def generate_data(u, seed=123, max_x=1, n=100, tau=3.333, theta=0.5):
    np.random.seed(seed)

    data = pd.DataFrame(np.random.uniform(0, 1, size=(n * 10, 4)),
                        columns=["p0", "x0", "p1", "x1"])
    for i in range(2):
        data[f"x{i}"] = data[f"x{i}"].values * max_x
    data = data[~((data.p0 >= data.p1) & (data.x0 >= data.x1))]
    data = data[~((data.p1 >= data.p0) & (data.x1 >= data.x0))]
    data = data.sample(n=n, replace=False)

    p0 = data.p0.values
    p1 = data.p1.values
    x0 = data.x0.values
    x1 = data.x1.values

    seu0 = p0 * u(x0, theta)
    seu1 = p1 * u(x1, theta)

    diff_eu = seu1 - seu0

    p_choice_1 = scipy.special.expit(tau * diff_eu)
    choices = np.zeros(n, dtype=int)
    choices[:] = p_choice_1 > np.random.random(size=n)
    data['choices'] = choices

    return data

# Likelihood whole model given M, $\theta_M$

def softplus(x):
    return np.log(1 + np.exp(x))


def objective(param, data, u_m):
    param = softplus(param)  # All parameters all supposed to be R+

    tau = param[0]
    theta = param[1]

    p0 = data.p0.values
    p1 = data.p1.values
    x0 = data.x0.values
    x1 = data.x1.values
    y = data.choices.values

    seu0 = p0 * u_m(x0, theta)
    seu1 = p1 * u_m(x1, theta)

    diff_eu = seu1 - seu0

    p_choice_1 = scipy.special.expit(tau * diff_eu)  # p choose 1
    p_choice_y = p_choice_1 ** y * (1 - p_choice_1) ** (1 - y)
    return - np.log(p_choice_y).sum()


def optimize(data, u_m=u_pow, x0=None):
    if x0 is None:
        x0 = (0.0, 0.0)  # Assume two parameters
    opt = scipy.optimize.minimize(objective, x0=x0, args=(data, u_m))
    return softplus(opt.x)


## Using Pymc3

def run_inference_pymc3(data, u_model, theta_model, tau_model):
    p0 = data.p0.values
    p1 = data.p1.values
    x0 = data.x0.values
    x1 = data.x1.values
    y = data.choices.values

    x = np.hstack((x0, x1))
    p = np.hstack((p0, p1))

    x_order = np.argsort(x)
    x_sorted = x[x_order]
    p_sorted = p[x_order]
    undo_sorted = np.argsort(x_order)

    X = x_sorted.reshape(-1, 1)
    uX = u_model(X, theta_model)
    factor = np.dot(uX, uX.T)

    n = len(y)

    with pm.Model() as model:
        length = pm.InverseGamma('length', 2, 2)
        var = pm.HalfCauchy('rho', 5)

        M = pm.gp.mean.Constant(uX.flatten())
        K = factor * var** 2 * pm.gp.cov.ExpQuad(1, length)

        gp = pm.gp.Latent(mean_func=M, cov_func=K)
        f = gp.prior("f", X=X)

        est_eu_sorted = p_sorted * f
        est_eu = est_eu_sorted[undo_sorted]
        est_diff_eu = est_eu[n:] - est_eu[:n]

        pm.Bernoulli('obs', logit_p=tau_model * est_diff_eu, observed=y)

    with model:
        trace = pymc3.sampling_jax.sample_tfp_nuts(10, tune=10,
                                                       chains=1,
                                                       target_accept=0.9)
#        trace = pm.sample(10, tune=10, chains=1, return_inferencedata=True, target_accept=0.90)
    return X, trace

def main():

    u_data = u_pow
    theta_data = 0.5
    tau_data = 3.333

    data = generate_data(u=u_data, tau=tau_data, theta=theta_data, n=1000,
                         seed=123)
    np.random.seed(12345)
    data = data.sample(n=100, replace=False)

    optimize(data, u_m=u_pow)


    u_model = u_pow
    theta_model = 0.5
    tau_model = 3.333

    X, trace = run_inference_pymc3(data=data, u_model=u_model,
                                   theta_model=theta_model, tau_model=tau_model)

    # f_samples = np.array(trace.posterior["f"])
    # f_samples = f_samples.reshape(f_samples.shape[0] * f_samples.shape[1], -1)
    #
    # fig, ax = plt.subplots()
    # plot_gp_dist(ax, f_samples, X)
    # ax.plot(X, u_model(X, theta=theta_model), ls="--");

if __name__ == "__main__":
    main()