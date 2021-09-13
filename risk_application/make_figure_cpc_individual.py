import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill
from tqdm import tqdm
import ntpath
import warnings


def plot_single_model(df_dm, idx, ax, fill_alpha=0.3):
    # Select data
    s = df_dm.subject.values[idx]
    is_s = df_dm.subject == s
    dm = df_dm.loc[is_s, "dm"].item()

    # Compute predictions, confidence, ... -----------
    test_x = torch.linspace(0.0, 1.0, 100)
    with warnings.catch_warnings():
        # this will suppress all warnings in this block
        warnings.simplefilter("ignore")
        m_pred, f_pred = dm.pred(test_x)
    test_x = test_x.numpy()
    m_pred = m_pred.numpy()
    f_pred = f_pred.numpy()
    f_mean = f_pred.mean(axis=0)
    lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

    # Plot -----------------------------------------
    conf = ax.fill_between(
        test_x, upper, lower, color='C0',
        alpha=fill_alpha,
        label="Model + correction - Confidence")
    mean, = ax.plot(test_x, f_mean, label="Model + correction - Mean")
    model, = ax.plot(test_x, m_pred, color='C1', ls="--", label="Model alone")

    return model, mean, conf,


def make_figure_model(df_dm):
    fig, axes = plt.subplots(ncols=10, nrows=13, figsize=(10, 12))
    axes_flatten = axes.flatten()
    for i in tqdm(range(125)):
        ax = axes_flatten[i]
        model, mean, conf, = plot_single_model(df_dm=df_dm, idx=i, ax=ax)
        # h_line, = ax.plot((0, 1), (0, 1), label="model")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in np.concatenate((axes[-1, :5], axes[-2, 5:])):
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_xlabel("reward")

    for ax in axes[:, 0]:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])
        ax.set_ylabel("utility")

    for ax in axes_flatten[125:]:
        ax.axis('off')

    fig.legend(
        handles=[model, mean, conf, ],
        bbox_to_anchor=(0.7, 0.065), loc="lower center",
        prop={'size': 11},
        bbox_transform=fig.transFigure, ncol=1)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    fig_name = f"fig/risk_cpc_individual.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()


def plot_single_loss(df_dm, idx, ax):

    # Select data
    s = df_dm.subject.values[idx]
    is_s = df_dm.subject == s
    dm = df_dm.loc[is_s, "dm"].item()

    # Plot -----------------------------------------
    ax.plot(dm.hist_loss)

    return np.min(dm.hist_loss), np.max(dm.hist_loss)


def make_figure_loss(df_dm):
    fig, axes = plt.subplots(ncols=10, nrows=13, figsize=(10, 12))
    axes_flatten = axes.flatten()

    min_, max_ = np.zeros(125), np.zeros(125)

    for i in tqdm(range(125)):
        ax = axes_flatten[i]
        min_[i], max_[i] = plot_single_loss(df_dm=df_dm, idx=i, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flatten:
        ax.set_ylim(min_.min(), max_.max())

    for ax in np.concatenate((axes[-1, :5], axes[-2, 5:])):
        ax.set_xlabel("loss")

    for ax in axes[:, 0]:
        ax.set_ylabel("iteration")

    for ax in axes_flatten[125:]:
        ax.axis('off')

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    fig_name = f"fig/risk_cpc_individual_loss.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()


def main():

    sns.set_context("paper")

    bkp_file = "bkp/dm_cpc_mean_correction=2_lr=05_epochs=1000_seed_cog_fit=12345_seed_dm_train=12345.pkl"

    np.random.seed(123)
    torch.manual_seed(123)

    # Loading
    df_dm = pd.read_pickle(bkp_file)
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))

    make_figure_model(df_dm)
    make_figure_loss(df_dm)


if __name__ == "__main__":
    main()
