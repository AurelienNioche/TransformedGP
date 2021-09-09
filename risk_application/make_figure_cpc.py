import os
import string
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill
from tqdm import tqdm
import ntpath

from accuracy.accuracy import accuracy_comparison_cpc


def main_plot(d_mean, d_unc, dm_low, dm_medium, dm_high, dm_uncertain,
              fig_name_ext):
    letters = (letter for letter in string.ascii_uppercase)

    # for left column only
    letter_placement_left = -0.22, 1.1
    letter_placement_right = -0.22, 1.13

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 3)

    gs_left = gs[0].subgridspec(3, 1)
    gs_right = gs[1:].subgridspec(2, 2)

    ax = fig.add_subplot(gs_left[0])

    sns.kdeplot(d_mean, ax=ax)
    x, y = ax.get_lines()[0].get_data()
    print(f"x-value mode KDE discr.={x[y.argmax()]:.3f}")
    ax.set_xlim(0, None)
    ax.set_xlabel("Discrepancy")

    ax.text(*letter_placement_left, s=next(letters), transform=ax.transAxes,
            fontsize=16,
            fontweight='bold', va='top', ha='right')

    ax = fig.add_subplot(gs_left[1])

    sns.kdeplot(d_unc, ax=ax)
    x, y = ax.get_lines()[0].get_data()
    print(f"x-value mode KDE discr. unc.={x[y.argmax()]:.3f}")
    ax.set_xlim(0, None)
    ax.set_xlabel("Discr. uncertainty")

    ax.text(*letter_placement_left, s=next(letters), transform=ax.transAxes,
            fontsize=16,
            fontweight='bold', va='top', ha='right')

    ax = fig.add_subplot(gs_left[2])

    sns.scatterplot(x=d_mean, y=d_unc, ax=ax)
    ax.set_xlabel("Discrepancy")
    ax.set_ylabel("Discr. uncert.")

    ax.text(*letter_placement_left, s=next(letters), transform=ax.transAxes,
            fontsize=16,
            fontweight='bold', va='top', ha='right')

    all_axes = [fig.add_subplot(gs_right[i, j]) for j in range(2) for i in
                range(2)]

    axes = (ax for ax in all_axes)

    for dm, dm_name in ((dm_low, "Human - Low discrepancy"),
                        (dm_medium, "Human - Moderate discr."),
                        (dm_high, "Human - High discrepancy"),
                        (dm_uncertain, "Human - Uncertain discr.")):
        ax = next(axes)
        ax.set_title(dm_name)

        fill_alpha = 0.3

        test_x = torch.linspace(0, 1, 100)
        m_pred, f_pred = dm.pred(test_x)

        test_x = test_x.numpy()
        m_pred = m_pred.numpy()
        f_pred = f_pred.numpy()

        f_mean = f_pred.mean(axis=0)
        lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

        letter = next(letters)
        ax.text(*letter_placement_right, s=letter, transform=ax.transAxes,
                fontsize=16,
                fontweight='bold', va='top', ha='right')

        # GP confidence
        ax.fill_between(test_x, upper, lower, color='C0',
                        alpha=fill_alpha,
                        label="Model + correction - Confidence")

        # GP mean
        ax.plot(test_x, f_mean, label="Model + correction - Mean")

        # Model
        ax.plot(test_x, m_pred, color='C1', ls="--",
                label="Model alone")

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.00])
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("reward")
        ax.set_ylabel("utility")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    fig_name = f"fig/risk_cpc{fig_name_ext}.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def accuracy_plot(df_dm, fig_name_ext):

    df_dm = accuracy_comparison_cpc(dataframe=df_dm,
                                    verbose=False)

    roc_m_all = df_dm.roc_m.values
    roc_dm_all = df_dm.roc_dm.values

    improvement = roc_dm_all - roc_m_all
    print(f"ROC-AUC improvement: "
          f"{np.nanmean(improvement)} +/= {np.nanstd(improvement)}SD")

    fig, axes = plt.subplots(nrows=2)

    ax = axes[0]
    sns.kdeplot(roc_m_all, ax=ax, label="Before corr.")
    sns.kdeplot(roc_dm_all, ax=ax, label="After corr.")
    ax.set_xlabel("ROC-AUC score")
    ax.legend()

    ax = axes[1]
    sns.scatterplot(x=roc_m_all, y=roc_dm_all, ax=ax)
    ax.plot((ax.get_xlim()[0], 1), (ax.get_xlim()[0], 1),
            ls="--", color="0.4", zorder=-3)
    ax.set_xlabel("ROC-AUC score before corr.")
    ax.set_xlabel("ROC-AUC score after corr.")

    fig.tight_layout()

    fig_name = f"fig/risk_cpc_accuray{fig_name_ext}.pdf"
    os.makedirs(ntpath.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"Created figure {fig_name}")


def data_processing(df_dm, bkp_file):
    if "d_mean" not in df_dm.columns:
        for s in tqdm(df_dm.subject.values):
            s_idx = df_dm[df_dm.subject == s].index.item()

            dm = df_dm.iloc[s_idx].dm

            test_x = torch.linspace(0.0, 1.00, 1000)
            m_pred, f_pred = dm.pred(test_x, n_samples=1000)

            m_pred = m_pred.numpy()
            f_pred = f_pred.numpy()

            f_mean = f_pred.mean(axis=0)
            lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

            d_mean_x = np.abs(f_mean - m_pred)
            d_unc_x = upper - lower

            df_dm.loc[df_dm.subject == s, "d_mean"] = d_mean_x.mean()
            df_dm.loc[df_dm.subject == s, "d_unc"] = d_unc_x.mean()

        # Save computed values
        loaded_dm = df_dm.dm.copy()
        df_dm.dm = df_dm.dm.apply(lambda x: dill.dumps(x))
        df_dm.to_pickle(bkp_file)
        df_dm.dm = loaded_dm

    d_mean = df_dm.d_mean.values
    d_unc = df_dm.d_unc.values

    theta = df_dm.d_unc.values

    print(f"Discrepancy (mean): min={d_mean.min()}, max={d_mean.max()}")
    print(f"Discrepancy (uncertainty): min={d_unc.min()}, max={d_unc.max()}")

    idx_uncertain = d_unc.argmax()
    dm_uncertain = df_dm.iloc[d_unc.argmax()].dm
    val_uncertain = d_unc[idx_uncertain]

    bound_unc = np.percentile(d_unc, [95, ], axis=0)[0]
    print(f"Bound uncertainty {bound_unc}")
    bounds_theta = np.percentile(theta, [2.5, 97.5], axis=0)
    print(f"Bounds theta {bounds_theta}")

    # low, medium and high discrepancy
    # select individuals with not extreme disc. uncertainty
    df_select = df_dm[(d_unc <= bound_unc)
                      & (theta >= bounds_theta[0])
                      & (theta <= bounds_theta[1])
                      ]  # & (df_dm.disc_mean < bound_mean), ]
    d_mean_slc = df_select.d_mean.values

    idx_low = d_mean_slc.argmin()
    dm_low = df_select.iloc[idx_low].dm
    val_low = d_mean_slc[idx_low]

    idx_high = d_mean_slc.argmax()
    dm_high = df_select.iloc[idx_high].dm
    val_high = d_mean_slc[idx_high]

    dist_to_middle_ground = np.abs(
        d_mean_slc - 0.5 * (d_mean_slc.max() - d_mean_slc.min()))
    dist_to_middle_ground[idx_high] = np.inf
    dist_to_middle_ground[idx_low] = np.inf
    idx_medium = np.argmin(dist_to_middle_ground)
    dm_medium = df_select.iloc[idx_medium].dm
    val_medium = d_mean_slc[idx_medium]

    print(f"Low: subject {df_select.iloc[idx_low].subject}; δ={val_low:.3f}")
    print(
        f"Medium: subject {df_select.iloc[idx_medium].subject}; δ={val_medium:.3f}")
    print(
        f"High: subject {df_select.iloc[idx_high].subject}; δ={val_high:.3f}")
    print(
        f"Uncertain: subject {df_dm.iloc[idx_uncertain].subject}; u_δ={val_uncertain:.3f}")

    return d_mean, d_unc, dm_low, dm_medium, dm_high, dm_uncertain


def create_figures(bkp_file, fig_name_ext=None):
    sns.set_context("paper")

    if fig_name_ext is None:
        fig_name_ext = ntpath.splitext(ntpath.basename(bkp_file))[0] \
            .replace("dm_artificial", "")

    np.random.seed(123)
    torch.manual_seed(123)

    # Loading
    df_dm = pd.read_pickle(bkp_file)
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))

    d_mean, d_unc, dm_low, dm_medium, dm_high, dm_uncertain = data_processing(df_dm, bkp_file)

    main_plot(
        d_mean=d_mean, d_unc=d_unc,
        dm_low=dm_low,
        dm_medium=dm_medium,
        dm_high=dm_high,
        dm_uncertain=dm_uncertain,
        fig_name_ext=fig_name_ext)

    accuracy_plot(df_dm=df_dm, fig_name_ext=fig_name_ext)


def main():

    bf = "bkp/dm_cpc_mean_correction=2_lr=05_epochs=1000_seed_cog_fit=12345_seed_dm_train=12345.pkl"
    create_figures(bkp_file=bf, fig_name_ext="")


if __name__ == "__main__":
    main()
