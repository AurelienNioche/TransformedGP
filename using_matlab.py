import os
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

import opt


# Seed
np.random.seed(123)

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)

REWARDS = "reward_only", "reward_and_certainty"

# Parameters
tau = 0.1
steps = np.linspace(0.01, 0.99, 10)
reward_f = REWARDS[0]


def plot_u(x_val):
    """
        Model-free utility function
    """

    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Plot function
    ax.plot(steps, x_val)

    # Plot identity
    ax.plot([0, 1], [0, 1], alpha=0.5, ls="--", color="black")

    # Pimp your plot
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('Optimal model-free utility function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coord = np.vstack((steps, x_val))
    ax.scatter(coord[0], coord[1], alpha=0.5)

    plt.savefig(f"{FIG_FOLDER}/u_fmincon_{reward_f}.png", dpi=200)
    plt.show()


def plot_prob_function(p_val):

    """
        Model-free probability function
    """
    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Plot function
    ax.plot(steps, p_val)

    # Plot identity
    ax.plot([0, 1], [0, 1], alpha=0.5, ls="--", color="black")

    # Pimp your plot
    ax.set_xlabel('p')
    ax.set_ylabel('w(p)')
    ax.set_title('Optimal model-free probability function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coord = np.vstack((steps, p_val))
    ax.scatter(coord[0], coord[1], alpha=0.5)

    plt.savefig(f"{FIG_FOLDER}/prob_fmincon_{reward_f}.png", dpi=200)
    plt.show()


def plot_softmax():

    # Softmax function given the difference of value between 2 options
    def f(v):
        return expit(v / tau)

    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = -1, 1
    y_min, y_max = 0, 1

    # Generate x-values
    x = np.linspace(x_min, x_max, 100)

    # Plot function
    ax.plot(x, f(x), ls=':', label='constraint', color="red")

    # Pimp your plot
    ax.set_xlabel('$EU(L_1) - EU(L_2)$')
    ax.set_ylabel('$P(L_1)$')
    ax.set_title('Softmax function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    plt.savefig(f"{FIG_FOLDER}/softmax_fmincon_{reward_f}.png", dpi=200)
    plt.show()


def matlab_objective(param):

    steps = np.linspace(0.01, 0.99, 10)

    task = pd.DataFrame(np.array(list(it.product(steps, repeat=4))),
                        columns=["p0", "x0", "p1", "x1"])

    task = task[~((task.p0 >= task.p1) & (task.x0 >= task.x1))]
    task = task[~((task.p1 >= task.p0) & (task.x1 >= task.x0))]
    task.reset_index(inplace=True, drop=True)

    p = np.vstack((task.p0.values, task.p1.values)).T
    x = np.vstack((task.x0.values, task.x1.values)).T
    n_trial, n_option = p.shape

    n_steps = len(steps)

    x_param = param[:n_steps]
    p_param = param[n_steps:]

    uf = {}
    pf = {}
    for i in range(n_steps):
        s = steps[i]
        uf[s] = x_param[i]
        pf[s] = p_param[i]

    su = np.zeros((n_trial, n_option))
    sp = np.zeros((n_trial, n_option))

    for i in range(n_trial):
        for j in range(n_option):

            x_ij = x[i, j]
            p_ij = p[i, j]

            su_ij = uf[x_ij]
            sp_ij = pf[p_ij]
            su[i, j] = su_ij
            sp[i, j] = sp_ij

    seu = sp * su

    p_choice1 = expit((seu[:, 1] - seu[:, 0]) / tau)
    rand = np.random.random(size=n_trial)
    c = np.zeros(n_trial, dtype=int)
    c[:] = p_choice1 > rand

    r = np.zeros(n_trial)
    for i in range(n_trial):

        ci = c[i]
        p_c = p[i, ci]
        x_c = x[i, ci]

        if reward_f == "reward_only":
            r[i] = p_c * x_c

        elif reward_f == "reward_and_certainty":
            h = - p_c * np.log(p_c) - (1-p_c) * np.log(1-p_c)
            r[i] = (p_c * x_c) * (1 - h)

        else:
            raise Exception

    return - r.sum()


def matlab_optimize():

    n_param = len(steps) * 2

    opt.start()

    x0 = np.hstack((steps, steps))
    lb = np.zeros(n_param)
    ub = np.ones(n_param)
    param, ll, exit_flat = opt.fmincon("using_matlab.matlab_objective",
                                       x0=x0, lb=lb, ub=ub)
    opt.stop()

    return param


def main():

    param = matlab_optimize()

    plot_u(param[:len(steps)])
    plot_prob_function(param[len(steps):])
    plot_softmax()


if __name__ == "__main__":
    main()
