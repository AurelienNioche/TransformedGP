import os
import pickle
import pystan
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import multiprocessing
import itertools as it
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

bkp_folder = 'bkp'
os.makedirs(bkp_folder, exist_ok=True)

np.random.seed(123)

def u(x, alpha):
    return x**(1 - alpha)

steps = np.linspace(0.01,0.99, 10)
task = pd.DataFrame(np.array(list(it.product(steps, repeat=4))), columns=["p0", "x0", "p1", "x1"])
task = task[~((task.p0 >= task.p1) & (task.x0 >= task.x1))]
task = task[~((task.p1 >= task.p0) & (task.x1 >= task.x0))]
task.reset_index(inplace=True, drop=True)

n_trial = len(task)

tau = 0.1
true_alpha = 0.4

seu0 = task.p0 * u(task.x0, true_alpha)
seu1 = task.p1 * u(task.x1, true_alpha)

p_choice_1 = scipy.stats.norm.cdf(seu1 - seu0)  # expit((seu1 - seu0)/tau)

choice = np.zeros(n_trial, dtype=int)
choice[:] = p_choice_1 > np.random.random(size=n_trial)
task["choice"] = choice

n = 100

p = np.concatenate((task.p0.values[:n], task.p1.values[:n])).T
x = np.concatenate((task.x0.values[:n], task.x1.values[:n])).T
y = task.choice.values[:n]
print("p", p)
print("x", x)
print("y", y)

model = """
data {
    int<lower=1> n_y;
    int y[n_y];
    int<lower=1> n_x;
    real x[n_x];
    real p[n_x];
    real tau;
}

transformed data {
}

parameters {
    vector<lower=0, upper=1>[n_x] u;
}

transformed parameters {
}

model {    
    vector[2] v;
    vector[2] p_choice;
    int c;
    real p_c;

    for (i in 1: n_y) {
        v[1] = p[i] * u[i];
        v[2] = p[n_y+i] * u[n_y+i];
        p_choice = softmax(v ./ tau);
        c = y[i]+1;
        p_c = p_choice[c];
        target += log(p_c);
    }

}
generated quantities {
}
"""

# Put it to true if you edit the model
force_compilation = False

# Where to save backup
bkp_folder = 'bkp'
os.makedirs(bkp_folder, exist_ok=True)
bkp_file = os.path.join(bkp_folder, 'gp_utility_naive.pkl')

if not os.path.exists(bkp_file) or force_compilation is True:

    # Compile the model
    sm = pystan.StanModel(model_code=model)

    # Save the model
    with open(bkp_file, 'wb') as f:
        pickle.dump(sm, f)
else:
    # Load the model
    sm = pickle.load(open(bkp_file, 'rb'))

data = {'n_x': len(x), 'n_y': len(y), 'x': x, 'y': y, 'p': p, 'tau': tau}

# Put it to true if you want to do a new run
force_run = False

# Where to save backup
bkp_file = os.path.join(bkp_folder, 'gp_utility_naive_fit.pkl')

if not os.path.exists(bkp_file) or force_run is True:

    # Train the model and generate samples
    fit_stan = sm.sampling(data=data,
                           iter=1000, chains=1, )  # algorithm="Fixed_param")

    # Save
    with open(bkp_file, 'wb') as f:
        pickle.dump(fit_stan, f)
else:
    # Load
    fit_stan = pickle.load(open(bkp_file, 'rb'))

print(fit_stan)