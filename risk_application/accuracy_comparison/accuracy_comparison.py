import numpy as np
import scipy.special
from sklearn.metrics import roc_auc_score
import torch
import pandas as pd
import dill


def compute_roc_auc(d, tau, u_func, theta=None):

    pA = d.p0.values
    pB = d.p2.values

    xA0 = d.x0.values
    xA1 = d.x1.values

    xB0 = d.x2.values
    xB1 = d.x3.values

    y = d.choices.values

    uxA0 = u_func(xA0, theta)
    uxA1 = u_func(xA1, theta)
    uxB0 = u_func(xB0, theta)
    uxB1 = u_func(xB1, theta)

    seuA = pA * uxA0 + (1 - pA) * uxA1
    seuB = pB * uxB0 + (1 - pB) * uxB1

    diff_seu = seuB - seuA

    logits = tau * diff_seu

    p_choice_B = scipy.special.expit(logits)
    p_choice_y = p_choice_B ** y * (1 - p_choice_B) ** (1 - y)

    log_prob = np.log(p_choice_y + np.finfo(float).eps)
    prob = np.exp(log_prob)

    try:
        roc_auc_d = roc_auc_score(y, prob)
    except ValueError as e:
        print(e)
        roc_auc_d = np.nan
    return roc_auc_d


def compute_accuracy_initial_model(dm):

    d = dm.data
    tau = dm.tau
    theta = dm.theta
    u = dm.u

    return compute_roc_auc(d=d, tau=tau, u_func=u, theta=theta)


# def compute_accuracy_corrected_model(dm, n_samples=10000):
#
#     d = dm.data
#     tau = dm.tau
#
#     u_func = lambda x, _: dm.pred(torch.from_numpy(x).float(),
#                                   n_samples=n_samples)[-1].numpy().mean(0)
#
#     return compute_roc_auc(d=d, tau=tau, u_func=u_func, theta=None)


def compute_accuracy_corrected_model(dm, n_samples=10000):

    d = dm.data
    y = d.choices.values

    dm.r_model.eval()
    with torch.no_grad():
        r_dist = dm.r_model(dm.train_x)
        log_prob = dm.expected_log_prob(function_dist=r_dist,
                                        observations=dm.train_y,
                                        n_samples=n_samples)
        prob = torch.exp(log_prob).numpy()

    try:
        roc_auc_d = roc_auc_score(y, prob)
    except ValueError as e:
        print(e)
        roc_auc_d = np.nan
    return roc_auc_d


def accuracy_comparison_single(dm):

    roc_auc_m = compute_accuracy_initial_model(dm=dm)
    roc_auc_dm = compute_accuracy_corrected_model(dm=dm)
    # if dataset == "artificial":
    # elif dataset == "cpc":
    #     roc_auc_dm = compute_accuracy_corrected_model_cpc(dm=dm)
    # else:
    #     raise ValueError
    return roc_auc_m, roc_auc_dm


def accuracy_comparison_cpc(dataframe=None, backup_file=None, verbose=True):

    np.random.seed(123)
    torch.manual_seed(123)

    if dataframe is None:

        assert backup_file is not None
        # Loading
        print(f"Loading {backup_file}")
        df_dm = pd.read_pickle(backup_file)
        df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))
        df_dm.index = df_dm.subject
        df_dm.drop("subject", axis=1, inplace=True)
    else:
        df_dm = dataframe

    i = 0
    for s in df_dm.index:

        dm = df_dm.loc[s, "dm"]

        roc_m, roc_dm = accuracy_comparison_single(dm)

        df_dm.loc[s, "roc_m"] = roc_m
        df_dm.loc[s, "roc_dm"] = roc_dm

        if verbose:
            print(f"{i} {s}: ROC initial = {roc_m:.3f}; ROC corr = {roc_dm:.3f}; improv. = {roc_dm - roc_m}")

        i += 1

    return df_dm


def accuracy_comparison_artificial(dataframe=None, backup_file=None):

    np.random.seed(123)
    torch.manual_seed(123)

    if dataframe is None:

        assert backup_file is not None
        # Loading
        print(f"Loading from {backup_file}...")
        df_dm = pd.read_pickle(backup_file)
        df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))
    else:
        df_dm = dataframe

    h_set = "sigmoid", "exp", "identity"
    u_set = "u_pow", "u_lin"

    for u_name in u_set:
        for h in h_set:

            dm = df_dm.loc[(u_name, h)].item()
            print(u_name, h, accuracy_comparison_single(dm))
