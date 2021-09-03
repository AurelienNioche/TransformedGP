import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dill
from tqdm.notebook import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from accuracy_comparison.accuracy_comparison import accuracy_comparison_single

sns.set_context("paper")


def plot_single(dm, s):
    test_x = torch.linspace(0.01, 0.99, 1000)

    fig, axes = plt.subplots(figsize=(8, 4), ncols=2)

    ax = axes[0]

    roc_m, roc_dm = accuracy_comparison_single(dm=dm, dataset="cpc")

    ax.set_title(
        f"subject {s} | roc_m={roc_m:.3f}; roc_dm={roc_dm:.3f} | "
        f"improv_roc={roc_dm - roc_m:.3f}")

    fill_alpha = 0.3

    m_pred, f_pred = dm.pred(test_x)

    test_x = test_x.numpy()
    m_pred = m_pred.numpy()
    f_pred = f_pred.numpy()

    f_mean = f_pred.mean(axis=0)
    lower, upper = np.percentile(f_pred, [2.5, 97.5], axis=0)

    # GP confidence
    ax.fill_between(test_x, upper, lower, color='C0',
                    alpha=fill_alpha,
                    label="Model + correction - Confidence")

    # GP mean
    ax.plot(test_x, f_mean, label="Model + correction - Mean")

    # Model
    ax.plot(test_x, m_pred, color='C1', ls="--", label="Model alone")

    ax.set_xlabel("reward")
    ax.set_ylabel("utility")

    ax = axes[1]
    ax.plot(dm.hist_loss)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")

    fig.tight_layout()
    return fig


def make_individual_plots(dataset=None, backup_file=None):
    assert backup_file is not None or dataset is not None

    if dataset is None:
        # Loading
        df_dm = pd.read_pickle(backup_file)
        df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))
        df_dm.index = df_dm.subject
        df_dm.drop("subject", axis=1, inplace=True)
    else:
        assert isinstance(dataset, pd.DataFrame)
        df_dm = dataset

    figs = []

    for s in tqdm(df_dm.index):
        dm = df_dm.loc[s, "dm"]
        fig = plot_single(dm=dm, s=s)
        figs.append(fig)

    fig_name = 'fig/individual_plots.pdf'
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    pp = PdfPages(fig_name)
    for fig in figs:
        pp.savefig(fig)
    pp.close()


def main():
    backup_file = "bkp/XXX"


if __name__ == "__main__":
    main()