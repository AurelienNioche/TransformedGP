import numpy as np
import matplotlib.pyplot as plt

import torch


def plot_results(
        dm, theta_truth=None, fill_alpha=0.3,
        x_max=2 ** 8,
        presentations=(64, 128, 192), title=None,
        ax=None):

    def create_input():

        delta = x_plot.copy()
        rep = np.zeros(x_plot.shape)

        for pres in presentations:

            # Already assume presentation at 0
            if pres == 0:
                continue

            idx_after = x_plot >= pres
            delta[idx_after] = x_plot[idx_after] - pres
            rep[idx_after] += 1

        x_input = np.concatenate((delta[None, :], rep[None, :]), axis=0).T
        return torch.from_numpy(x_input.astype(np.float32))

    x_plot = np.linspace(0, x_max, 500)
    test_x = create_input()
    m_pred, f_pred = dm.pred(test_x)

    test_x = test_x.numpy()
    m_pred = m_pred.numpy()
    f_pred = f_pred.numpy()

    f_mean = f_pred.mean(axis=0)
    lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # GP confidence
    h_conf = ax.fill_between(x_plot, upper, lower, color='C0',
                             alpha=fill_alpha,
                             label="Model + correction - Confidence")

    # GP mean
    h_mean, = ax.plot(x_plot, f_mean, label="Model + correction - Mean")

    # Model
    h_model, = ax.plot(x_plot, m_pred, color='C1', ls="--",
                       label="Model alone")

    if theta_truth is not None:
        # Ground truth
        truth = dm.m(test_x, theta_truth)
        h_truth, = ax.plot(x_plot, truth, color='C2', ls="--",
                           label="Ground truth")

    ax.set_xlabel("time")
    ax.set_ylabel("probability of recall")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_losses(hist_loss, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(hist_loss)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    if title is not None:
        ax.set_title(title)
    return ax
