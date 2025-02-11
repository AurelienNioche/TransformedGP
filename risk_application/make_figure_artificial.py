import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill
import string
import ntpath

from cognitive_modeling.models.utility_models import u_pow, u_lin


def create_main_fig(df_dm, h_set, u_set, u_truth, theta_truth, fig_name_ext):

    nrows = len(h_set)
    ncols = len(u_set)*2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 5))

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
            f_pred = f_pred.numpy()

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
        bbox_to_anchor=(0.48, -0.09), loc="lower center",
        prop={'size': 11},
        bbox_transform=fig.transFigure, ncol=3)

    for axes_ in axes[:, 1::2]:
        for ax in axes_:
            ax.set_ylim(err_min, err_max)

    for axes_ in axes[:, 0::2]:
        for ax in axes_:
            ax.set_ylim(-0.25, 1.25)

    fig.text(0.1, 0.99, "Assuming the adequate utility model", fontsize=12,
             transform=fig.transFigure)

    fig.text(0.56, 0.99, "Assuming a linear utility model instead",
             fontsize=12, transform=fig.transFigure)

    fig.text(-0.01, 0.84, "h: sigmoid", fontsize=14, transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.text(-0.01, 0.52, "h: exp", fontsize=14, transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.text(-0.01, 0.19, "h: identity", fontsize=14,
             transform=fig.transFigure,
             verticalalignment='center', rotation=90)

    fig.tight_layout()

    fig_name = f"fig/risk_artificial{fig_name_ext}.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def create_loss_fig(df_dm, h_set, u_set, fig_name_ext):

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

    fig_name = f"fig/risk_artificial_loss{fig_name_ext}.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def compute_statistics(df_dm, u_set, h_set, u_truth, theta_truth):
    for u in u_set:
        for h in h_set:

            dm = df_dm.loc[(u.__name__, h)].item()

            test_x = torch.linspace(0.0, 1.0, 1000)
            m_pred, f_pred = dm.pred(test_x, n_samples=1000)

            truth = u_truth(test_x.numpy(), theta_truth)

            m_pred = f_pred.numpy()
            f_pred = f_pred.numpy()

            f_mean = f_pred.mean(axis=0)
            lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

            d_mean_x = np.abs(f_mean - m_pred)
            err_x = np.abs(f_mean - truth)
            d_unc_x = upper - lower

            d_mean = d_mean_x.mean()
            d_unc = d_unc_x.mean()
            err = err_x.mean()
            print(f"{u.__name__} {h}: d_mean={d_mean:.3f}; d_unc={d_unc:.3f}; {err:.3f}")


def create_figures(bkp_file, fig_name_ext=None):

    if fig_name_ext is None:
        fig_name_ext = ntpath.splitext(ntpath.basename(bkp_file))[0]\
            .replace("dm_artificial", "")

    sns.set_context("paper")

    u_truth = u_pow
    theta_truth = 0.5
    h_set = "sigmoid", "exp", "identity"
    u_set = u_pow, u_lin

    print(f"Loading from {bkp_file}...")
    df_dm = pd.read_pickle(bkp_file)
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))

    compute_statistics(df_dm=df_dm, h_set=h_set, u_set=u_set,
                       u_truth=u_truth, theta_truth=theta_truth)

    create_main_fig(df_dm=df_dm, h_set=h_set, u_set=u_set,
                    u_truth=u_truth, theta_truth=theta_truth,
                    fig_name_ext=fig_name_ext)

    create_loss_fig(df_dm=df_dm, h_set=h_set, u_set=u_set,
                    fig_name_ext=fig_name_ext)


def main():
    bkp_file = "bkp/dm_artificial_mean_cor=2_lr=05_epochs=1000_seed_data=12345_seed_cog_fit=12345_seed_dm_train=12345.pkl"
    create_figures(bkp_file=bkp_file, fig_name_ext="")


if __name__ == "__main__":
    main()
