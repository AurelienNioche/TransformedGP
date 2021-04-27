#Reproduce or fix Aureliens code
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

stan_model = None

def generate_data(u, seed=1234):
    np.random.seed(seed)
    N_pool = 8000
    task = pd.DataFrame(np.random.uniform(-2, 2, size=(N_pool, 4)), columns=["p0", "x0", "p1", "x1"])
    task["p0"] = np.random.uniform(0,1, N_pool)
    task["p1"] = np.random.uniform(0,1, N_pool)
    task = task[~((task.p0 >= task.p1) & (task.x0 >= task.x1))]
    task = task[~((task.p1 >= task.p0) & (task.x1 >= task.x0))]

    seu0 = task.p0 * u(task.x0)
    seu1 = task.p1 * u(task.x1)

    diff_eu = seu1 - seu0
    task['diff_eu'] = diff_eu

    p_choice_1 = expit(1.0*diff_eu) #this is probability of choosing 1 
    choice = np.zeros(p_choice_1.shape, dtype=int)
    choice[:] = p_choice_1 > np.random.uniform(size=choice.shape)
    task["choice"] = choice

    n = 550
    task = task.sample(n=n, replace=False)
    #task = task.sort_values(by="diff_eu")
    #task.reset_index(inplace=True, drop=True)

    p0 = task.p0.values
    p1 = task.p1.values
    x0 = task.x0.values
    x1 = task.x1.values
    choices = task.choice.values
    diff_eu = task.diff_eu.values

    return p0, x0, p1, x1, choices

beta_real = 2.71
u_data = lambda x: 1 - np.exp(-beta_real*x)
u_data = lambda x: x
xx = np.arange(-1,1, 0.02)
plt.plot(xx, u_data(xx))
plt.hlines(0, -1, 1)
plt.vlines(0, -10, 3)
plt.show()

p0, x0, p1, x1, choices = generate_data(u_data, seed = np.random.randint(200))

#introduce things regarded to GP
l = 1.5
sigmasq = 1.0
N_eval = 26
x_eval = np.linspace(-3,3, N_eval)

import pystan
if stan_model is None:
    stan_model = pystan.StanModel(file="ratio_discrepancy.stan")
prior_shape = 1.5
prior_rate = 0.99
stan_data = {"N": len(p0),
             "N_eval": N_eval,
             "p0": p0,
             "x0": x0,
             "p1": p1,
             "x1": x1,
             "chose1": choices,
             "prior_shape": prior_shape,
             "prior_rate": prior_rate,
             "x_eval": x_eval,
            #  "l": l,
            #  "sigmasq": sigmasq,
            #  "beta": beta_real/2,
             "jitter": 1e-8}

samples = stan_model.sampling(data = stan_data, verbose=True, chains=4)
print(samples)

import arviz as az
import arviz.labels as azl
samples_az = az.from_pystan(samples)
#az.plot_pair(samples_az, var_names=["beta", "r"], coords={"r_dim_0":[1,2]}, divergences=True)
#az.plot_pair(samples_az, var_names=["r"], coords={"r_dim_0":[0,1,2,3]}, divergences=True)
az.plot_pair(samples_az, var_names=["beta","r", "l", "sigmasq"], coords={"r_dim_0":[4]}, divergences=True)
#az.plot_trace(samples, "beta", divergences=True)
#az.plot_trace(samples, "l", divergences=True)
#az.plot_trace(samples, "sigmasq", divergences=True)
#az.plot_trace(samples, "r", divergences=True, coords={"r_dim_0":[0]})
#plt.savefig("asd.jpg")
softplus = lambda x: np.log(np.exp(x) + 1.)
softplus_inv = lambda x: np.log(np.exp(x)-1.)#not sure about this
r_mean = samples["r"].mean(axis=0)
r_median = np.median(samples["r"], axis=0)
az.style.use("arviz-darkgrid")
#az.plot_hdi(x_eval, samples["r"], color="k")
az.plot_hdi(x_eval, samples["r"], color="k", hdi_prob=0.75)
plt.plot(x_eval, r_mean, label="mean")
plt.plot(x_eval, r_median, color="cyan", label="median")
plt.plot(x_eval, samples["r"][np.random.choice(100, 1).item()], label="single sample")
plt.legend()

plt.hist(samples["r"][:,15])

import torch
xxx = torch.arange(0, 4, 0.1)
prior_dist = torch.distributions.Gamma(torch.tensor(prior_shape), torch.tensor(prior_rate))
plt.hist(samples["beta"], density=True)
plt.plot(xxx, prior_dist.log_prob(xxx).exp())
plt.vlines(beta_real, 0, 4)
plt.show()

print(np.mean(samples["beta"]))
print(np.median(samples["beta"]))

xx = np.arange(-3,3, 0.05)
u_generator = lambda beta: (lambda x: 1 - np.exp(-beta*x))
plt.plot(xx, u_data(xx), color="green", linewidth=3, zorder= 100, label="generating")
plt.hlines(0, -1, 1)
plt.vlines(0, np.min(u_data(xx)), np.max(u_data(xx)))
#GP related
import scipy.spatial.distance as dlib
dist_eval = dlib.pdist(x_eval.reshape(-1, 1))
dist_eval = dlib.squareform(dist_eval)
dist_pred = dlib.cdist(xx.reshape(-1,1), x_eval.reshape(-1, 1))
squared_exp = lambda dd, lengthscale, sigmasq: sigmasq * np.exp(-0.5 * dd**2 / lengthscale**2)
jitter = 1e-8
I_jitter = jitter * np.eye(N_eval)
#plot
u_mean = np.zeros(len(xx))
N_samples = 350
for ii in range(N_samples):
    #theoretical model
    try:
        M_sample = u_generator(samples["beta"][ii])
    except:
        M_sample = u_generator(beta_real)
    #covariances
    l_sample = samples["l"][ii]
    sigmasq_sample = samples["sigmasq"][ii]
    K_eval = squared_exp(dist_eval, l_sample, sigmasq_sample) + I_jitter
    K_pred = squared_exp(dist_pred, l_sample, sigmasq_sample)
    #other stuff
    r_sample = samples["r"][ii]#todo inverse map?
    r_sample_unc = softplus_inv(r_sample)
    r_pred_unc = 0.5 + K_pred @ np.linalg.inv(K_eval) @ (r_sample_unc - 0.5)#assume 1.0 mean
    r_pred = softplus(r_pred_unc)
    ux = M_sample(xx) * r_pred
    u_mean += ux #accumulate mean
    plt.plot(xx, ux, color="red", alpha=0.2)
    # plt.plot(x_eval, softplus_inv(samples["r"]).mean(axis=0), color="orange", linewidth=4)
    # plt.plot(x_eval, r_sample_unc, color="red", alpha=0.1)
    # plt.plot(xx, r_pred_unc, color="red", alpha=0.1)

u_mean = u_mean / N_samples
plt.plot(xx, u_mean, color="orange", linewidth=3, label="recovered (mean)")
plt.ylim(-5,2)
plt.legend()