import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

import opt


# Seed
np.random.seed(123)



def plot_u(x_steps, x_val):

    # Model-free utility function

    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Plot estimate
    ax.plot(x_steps, x_val)

    ax.plot([0, 1], [0, 1], alpha=0.5, ls="--")

    # Pimp your plot
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('Optimal model-free utility function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coord = np.vstack((x_steps, x_val))
    ax.scatter(coord[0], coord[1], alpha=0.5)

    plt.show()


def plot_prob_function(p_steps, p_val):

    # Model-free probability function

    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Plot estimate
    ax.plot(p_steps, p_val)

    ax.plot([0, 1], [0, 1], alpha=0.5, ls="--")

    # Pimp your plot
    ax.set_xlabel('p')
    ax.set_ylabel('w(p)')
    ax.set_title('Optimal model-free probability function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coord = np.vstack((p_steps, p_val))
    ax.scatter(coord[0], coord[1], alpha=0.5)

    plt.show()


def plot_softmax(tau):

    # Softmax function given the difference of value between 2 options
    def f(v, tau):
        return expit(v / tau)


    # Create fig
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits
    x_min, x_max = -1, 1
    y_min, y_max = 0, 1

    # Generate x-values
    x = np.linspace(x_min, x_max, 100)

    # Plot estimate
    # ax.plot(x, f(x, tau_est), label="fit")

    # Plot truth
    ax.plot(x, f(x, tau), ls=':', label='constraint', color="red")

    # Pimp your plot
    ax.set_xlabel('$EU(L_1) - EU(L_2)$')
    ax.set_ylabel('$P(L_1)$')
    ax.set_title('Softmax function')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    plt.show()


def matlab_objective(param):

    steps = np.linspace(0.01, 0.99, 10)
    tau = 0.11
    reward_f = "reward_and_certainty"

    task = pd.DataFrame(np.array(list(it.product(steps, repeat=4))),
                        columns=["p0", "x0", "p1", "x1"])

    task = task[~((task.p0 > task.p1) & (task.x0 > task.x1))]
    task = task[~((task.p1 > task.p0) & (task.x1 > task.x0))]
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
            su_ij = uf[x[i, j]]
            sp_ij = pf[p[i, j]]
            su[i, j] = su_ij
            sp[i, j] = sp_ij

    seu = sp * su

    p_choice = np.exp(seu.T / tau)
    p_choice /= p_choice.sum(axis=0)
    p_choice = p_choice.T

    r = np.zeros(n_trial)
    for i in range(n_trial):

        # c = int(p_choice[1] > p_choice[0])
        c = np.random.choice(np.arange(n_option), p=p_choice[i])

        p_c = p[i, c]
        x_c = x[i, c]

        if reward_f == "reward_only":
            r[i] = p_c * x_c

        elif reward_f == "reward_and_certainty":
            h = - p_c * np.log(p_c) - (1-p_c) * np.log(1-p_c)
            r[i] = (p_c * x_c) * (1 - h)

        else:
            raise Exception

    return - r.sum() * 1000


def matlab_optimize(steps):

    n_param = len(steps) * 2

    opt.start()

    x0 = np.hstack((steps, steps))
    lb = np.zeros(n_param)
    ub = np.ones(n_param)
    param, ll, exit_flat = opt.fmincon("tamere.matlab_objective",
                                       x0=x0, lb=lb, ub=ub)
    opt.stop()

    return param


def main():

    steps = np.linspace(0.01, 0.99, 10)

    param = matlab_optimize(steps)

    plot_u(x_steps=steps, x_val=param[:len(steps)])
    plot_prob_function(p_steps=steps, p_val=param[len(steps):])


if __name__ == "__main__":
    main()
