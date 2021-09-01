import numpy as np
import scipy.special
from sklearn.metrics import roc_auc_score
import torch


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

    roc_auc_d = roc_auc_score(y, prob)
    return roc_auc_d


def compute_accuracy_initial_model(dm):

    d = dm.data
    tau = dm.tau
    theta = dm.theta
    u = dm.u

    return compute_roc_auc(d=d, tau=tau, u_func=u, theta=theta)


def compute_accuracy_corrected_model(dm):

    d = dm.data
    tau = dm.tau

    u_func = lambda x, _: dm.pred(torch.from_numpy(x).float(), n_samples=10000)[-1].numpy().mean(0)

    return compute_roc_auc(d=d, tau=tau, u_func=u_func, theta=None)


def accuracy(dm):

    roc_auc_dm = compute_accuracy_corrected_model(dm=dm)
    roc_auc_m = compute_accuracy_initial_model(dm=dm)
    return roc_auc_m, roc_auc_dm




