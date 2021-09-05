import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import dill
import string
import torch
import ntpath

from data.data import get_data


def main_plot(data, df_dm):

    # Initialize plot
    fig, axes = plt.subplots(figsize=(8, 4), nrows=2, ncols=3)

    axis_labels_fontsize = 11

    err_max = - np.inf
    err_min = np.inf
    h_max = - np.inf
    h_min = np.inf

    letter_idx = 0
    letters = (letter for letter in string.ascii_uppercase)

    for c_idx, m_name in enumerate(("Linf", "Logf", "Jpps")):

        dm = df_dm.loc[m_name, "dm"]
        title = m_name.upper()

        # extract data
        train_x = data.age.values
        train_y = data.height.values

        # Pickup testing points
        test_x = torch.linspace(0, train_x.max(), 50).double()

        # Get mean, lower and upper confidence bounds
        m_pred, gp_mean, gp_lower, gp_upper = dm.pred(test_x)

        # Convert to numpy for plot
        test_x = test_x.numpy()

        # -------------------------------------------------------------------- #

        ax = axes[0, c_idx]

        # Plot training data as black stars
        h_obs, = ax.plot(train_x, train_y, 'x', label='Observations',
                         alpha=0.05, color='black')
        # Plot predictive means as blue line
        h_mean, = ax.plot(test_x, gp_mean, 'C0',
                          label="Model + correction - Mean")
        # Shade between the lower and upper confidence bounds
        h_conf = ax.fill_between(test_x, gp_lower, gp_upper, alpha=0.3,
                                 label="Model + correction - Confidence")

        # Plot model predictions
        h_model, = ax.plot(test_x, m_pred, color='C1', ls="--",
                           label="Model alone")

        if c_idx == 0:
            ax.set_ylabel("height", fontsize=axis_labels_fontsize)
        else:
            ax.set_yticklabels([])

        ax.set_title(title)

        ax.set_xticks([])

        ax.text(0.05, 1.2, next(letters), transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='right')
        letter_idx += 1

        y_lim = ax.get_ylim()
        h_min = min(h_min, y_lim[0])
        h_max = max(h_max, y_lim[1])

        # -------------------------------------------------------------------- #

        ax = axes[1, c_idx]

        # Add horizontal line
        ax.hlines(y=0, ls="--", xmin=test_x.min(), xmax=test_x.max(),
                  color="black")

        # Compute correction
        corr = gp_mean - m_pred
        corr_lower = gp_lower - m_pred
        corr_upper = gp_upper - m_pred

        # Plot correction and CI
        h_corr_mean, = ax.plot(test_x, corr, color='C3',
                               label="Discrepancy - Mean")
        h_corr_conf = ax.fill_between(test_x, corr_lower, corr_upper,
                                      alpha=0.3,
                                      label="Discrepancy - Confidence",
                                      color='C3', )

        # Add legend / title / format axis
        ax.set_xlabel("age", fontsize=axis_labels_fontsize)

        y_lim = ax.get_ylim()
        err_min = min(err_min, y_lim[0])
        err_max = max(err_max, y_lim[1])

        if c_idx == 0:
            ax.set_ylabel("discrepancy (height)",
                          fontsize=axis_labels_fontsize)
        else:
            ax.set_yticklabels([])

    # -------------------------------------------------------------------- #

    for c_idx in range(3):
        axes[0, c_idx].set_ylim(h_min, h_max)
        axes[1, c_idx].set_ylim(err_min, err_max)

    for ax in axes.flatten():
        ax.set_xlim(data.age.min() - 0.1, data.age.max() + 0.1)

    fig.legend(handles=[h_model, h_corr_mean, h_mean, h_corr_conf, h_conf],
               bbox_to_anchor=(0.5, -0.12), loc="lower center",
               prop={"size": 10.5},
               bbox_transform=fig.transFigure, ncol=3)

    fig.tight_layout()

    fig_name = "fig/growth.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def teaser_plot(df_dm, data):

    # Initialize plot
    fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)

    alpha_star = 0.05

    m_name = "Logf"

    dm = df_dm.loc[m_name, "dm"]

    h_min = np.inf
    h_max = - np.inf

    # extract data
    train_x = data.age.values
    train_y = data.height.values

    # Pickup testing points
    test_x = torch.linspace(0, train_x.max(), 50).double()

    # Get mean, lower and upper confidence bounds
    m_pred, gp_mean, gp_lower, gp_upper = dm.pred(test_x)

    # Convert to numpy for plot
    test_x = test_x.numpy()

    # ---------------------------------------------------------- #

    ax = axes[0]

    # Plot training data as black stars
    h_obs, = ax.plot(train_x, train_y, 'x', label='Observations',
                     alpha=alpha_star, color='black')

    # Plot model predictions
    h_model, = ax.plot(test_x, m_pred, color='C1', ls="--",
                       label="Model alone")

    # Add legend / title / format axis
    ax.legend(handles=[h_obs, h_model, ], )
    ax.set_xlabel("age")
    ax.set_ylabel("height")

    y_lim = ax.get_ylim()
    h_min = min(h_min, y_lim[0])
    h_max = max(h_max, y_lim[1])

    # ---------------------------------------------------------- #

    ax = axes[1]

    # Plot training data as black stars
    h_obs, = ax.plot(train_x, train_y, 'x', label='Observed data',
                     alpha=alpha_star, color='black')
    # Plot predictive means as blue line
    h_mean, = ax.plot(test_x, gp_mean, 'C0', label="Model + correction - Mean")
    # Shade between the lower and upper confidence bounds
    h_conf = ax.fill_between(test_x, gp_lower, gp_upper, alpha=0.3,
                             label="Model + correction - Confidence")

    # Plot model predictions
    h_model, = ax.plot(test_x, m_pred, color='C1', ls="--",
                       label="Model alone")

    # Add legend / title / format axis
    ax.legend(handles=[h_obs, h_model, h_mean, h_conf], )
    ax.set_xlabel("age")
    ax.set_ylabel("height")

    y_lim = ax.get_ylim()
    h_min = min(h_min, y_lim[0])
    h_max = max(h_max, y_lim[1])

    # --------------------------------------------------------- #

    ax = axes[2]

    # Add horizontal line
    ax.hlines(y=0, ls="--", xmin=test_x.min(), xmax=test_x.max(),
              color="black")

    # Compute correction
    corr = gp_mean - m_pred
    corr_lower = gp_lower - m_pred
    corr_upper = gp_upper - m_pred

    # Plot correction and CI
    h_corr_mean, = ax.plot(test_x, corr, color='C3',
                           label="Discrepancy - Mean")
    h_corr_conf = ax.fill_between(test_x, corr_lower, corr_upper,
                                  alpha=0.3,
                                  label="Discrepancy - Confidence",
                                  color='C3', )

    # Add legend / title / format axis
    ax.set_xlabel("age")
    ax.set_ylabel("discrepancy (height)")
    ax.legend(handles=[h_corr_mean, h_corr_conf])

    # -------------------------------------------------------- #

    for i, ax in enumerate(axes):
        if i < 2:
            ax.set_ylim(h_min, h_max)
        ax.set_xlim(data.age.min() - 0.2, data.age.max() + 0.2)

    fig.tight_layout()

    fig_name = "fig/teaser.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def main():
    sns.set_context("paper")

    data = get_data()

    df_dm = pd.read_pickle("bkp/dm_growth.pkl")
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))
    df_dm.index = df_dm.m
    df_dm.drop("m", axis=1, inplace=True)

    # Short analysis
    for c_idx, m_name in enumerate(("Linf", "Logf", "Jpps")):
        dm = df_dm.loc[m_name].dm

        train_x = torch.from_numpy(data.age.values).double()
        test_x = torch.linspace(train_x.min(), train_x.max(), 10000).double()

        # Get mean, lower and upper confidence bounds
        m_pred, f_mean, f_lower, f_upper = dm.pred(test_x)

        d_x = f_mean - m_pred
        d_unc_x = f_upper - f_lower

        d_mean = np.abs(d_x).mean()
        d_unc = d_unc_x.mean()
        print(f"{m_name}: d_mean={d_mean:.3f}; d_unc={d_unc:.3f}")

    main_plot(data=data, df_dm=df_dm)
    teaser_plot(data=data, df_dm=df_dm)


if __name__ == "__main__":
    main()


