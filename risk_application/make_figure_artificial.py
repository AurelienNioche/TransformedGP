import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill
import string

from cognitive_modeling.models.utility_models import u_pow, u_lin


def create_main_fig(df_dm, h_set, u_set, u_truth, theta_truth, seed):
    nrows = len(h_set)
    ncols = len(u_set)*2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 10))

    err_max = - np.inf
    err_min = np.inf

    letters = (letter for letter in string.ascii_uppercase)

    for h_idx, h in enumerate(h_set):

        row_idx = h_idx

        for u_idx, u in enumerate(u_set):
            col_idx = u_idx * 2

            dm = df_dm.loc[(u.__name__, h)].item()

            fill_alpha = 0.3

            test_x = torch.linspace(0, 1, 100)
            m_pred, f_pred = dm.pred(test_x)

            test_x = test_x.numpy()
            m_pred = m_pred.numpy()
            f_pred = f_pred.detach().numpy()

            truth = u_truth(test_x, theta_truth)

            f_mean = f_pred.mean(axis=0)
            lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

            ax0, ax1 = axes[row_idx, col_idx:col_idx + 2]

            ax = ax0

            ax.text(-0.1, 1.15, next(letters), transform=ax.transAxes,
                    fontsize=16,
                    fontweight='bold', va='top', ha='right')

            # GP confidence
            h_conf = ax.fill_between(test_x, upper, lower, color='C0',
                                     alpha=fill_alpha,
                                     label="Model + correction - Confidence")

            # GP mean
            h_mean, = ax.plot(test_x, f_mean,
                              label="Model + correction - Mean")

            # Model
            h_model, = ax.plot(test_x, m_pred, color='C1', ls="--",
                               label="Model alone")

            # Ground truth
            h_truth, = ax.plot(test_x, truth, color='C2', ls="--",
                               label="Ground truth")

            ax.set_xlabel("reward")
            ax.set_ylabel("utility")

            # ax.legend(handles=[h_model, h_mean, h_conf])

            ax = ax1

            corr = f_mean - m_pred
            corr_lower = lower - m_pred
            corr_upper = upper - m_pred

            # Error confidence
            corr_conf = ax.fill_between(test_x, corr_upper, corr_lower,
                                        color='C3', alpha=fill_alpha,
                                        label="Discrepancy - Confidence")

            # Error mean
            corr_mean, = ax.plot(test_x, corr,
                                 color="C3",
                                 label="Discrepancy - Mean")

            # Baseline
            ax.plot(test_x, torch.zeros(m_pred.shape[0]), color='black',
                    ls="--")

            ax.set_xlabel("reward")
            ax.set_ylabel("discrepancy (utility)")

            y_lim = ax.get_ylim()
            err_min = min(err_min, y_lim[0])
            err_max = max(err_max, y_lim[1])

    fig.legend(
        handles=[h_model, corr_mean, h_mean, corr_conf, h_conf, h_truth],
        bbox_to_anchor=(0.5, -0.05), loc="lower center",
        bbox_transform=fig.transFigure, ncol=3)

    for axes_ in axes[:, 1::2]:
        for ax in axes_:
            ax.set_ylim(err_min, err_max)

    for axes_ in axes[:, 0::2]:
        for ax in axes_:
            ax.set_ylim(-0.25, 1.25)

    fig.text(0.18, 0.99, "Assuming the adequate utility model ('pow')",
             fontsize=14, transform=fig.transFigure,
             verticalalignment='center')

    fig.text(0.6, 0.99, "Assuming a linear utility model (instead of 'pow')",
             fontsize=14, transform=fig.transFigure,
             verticalalignment='center')

    fig.text(-0.01, 0.82, "h: sigmoid", fontsize=14, transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.text(-0.01, 0.5, "h: exp", fontsize=14, transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.text(-0.01, 0.18, "h: identity", fontsize=14,
             transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.tight_layout()

    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/risk_artificial_seed={seed}.pdf", bbox_inches='tight')


def create_loss_fig(df_dm, h_set, u_set, seed):
    nrows = len(h_set)
    ncols = len(u_set)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10))

    letters = (letter for letter in string.ascii_uppercase)

    for row_idx, h in enumerate(h_set):

        for col_idx, u in enumerate(u_set):
            dm = df_dm.loc[(u.__name__, h)].item()

            ax = axes[row_idx, col_idx]

            ax.text(-0.1, 1.15, next(letters), transform=ax.transAxes,
                    fontsize=16,
                    fontweight='bold', va='top', ha='right')

            ax.plot(dm.hist_loss)
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss")

    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/risk_artificial_loss_seed={seed}.pdf",
                bbox_inches='tight')


def main(seed, use_mean_correction):
    sns.set_context("paper")

    u_truth = u_pow
    theta_truth = 0.5
    h_set = "sigmoid", "exp", "identity"
    u_set = u_pow, u_lin

    # Loading
    bkp_file = f"bkp/dm_artificial{'_mean_corrected' if use_mean_correction else ''}_seed={seed}.pkl"
    print(f"Loading from {bkp_file}...")
    df_dm = pd.read_pickle(bkp_file)
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))

    create_main_fig(df_dm=df_dm, h_set=h_set, u_set=u_set,
                    u_truth=u_truth, theta_truth=theta_truth, seed=seed)

    create_loss_fig(df_dm=df_dm, h_set=h_set, u_set=u_set, seed=seed)


if __name__ == "__main__":
    for seed in (1, 12, 123, 12345, 123456, 1234567):
        main(seed=seed, use_mean_correction=True)
