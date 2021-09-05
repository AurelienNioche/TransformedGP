import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
from typing import Callable, Union

import torch

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky

import git

REPO = git.Repo(search_parent_directories=True)


class GPClassificationModel(ApproximateGP):

    def __init__(self, inducing_points: torch.Tensor,
                 learn_inducing_locations: bool):

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = \
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


def safe_log(x): return torch.log(x + 1e-07)


def identity(x): return x


def sigmoid_second_derivative(x, ceil_exp=50.0):
    y = torch.zeros_like(x)
    under_ceil = x < ceil_exp
    e_x = torch.exp(x[under_ceil])
    y[under_ceil] = - ((e_x - 1) * e_x) / (e_x + 1)**3
    return y


class DiscrepancyModel:

    git_branch = REPO.active_branch.name
    git_hash = REPO.head.commit.hexsha

    def __init__(
            self,
            data: pd.DataFrame,
            u: Callable,
            theta: Union[int, float],
            h: Union[str, Callable],
            tau: Union[int, float],
            inducing_points: Union[np.ndarray, torch.Tensor, list],
            learn_inducing_locations: bool = False,
            jitter: Union[int, float] = 1e-07,
            n_samples: int = 40,
            cholesky_max_tries: int = 1000,
            mean_correction: int = 0):

        self.data = data

        self.u = u
        self.theta = theta
        self.tau = tau

        self.n_samples = n_samples

        self.jitter = jitter
        self.cholesky_max_tries = cholesky_max_tries
        self.mean_correction = mean_correction

        self.h, self.h_inv, self.h_second = self.set_h_functions(h)

        self.train_x, self.train_y, self.train_p, self.train_y, \
            self.n_output_total, self.n_output_per_lot \
            = self.format_data(data)

        self.n_x = self.train_x.size(0)
        self.n_y = self.train_y.size(0)

        self.h_inv_m = self.h_inv(u(self.train_x, self.theta))

        inducing_points = self.set_inducing_points(inducing_points)

        self.r_model = GPClassificationModel(
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations)

        self.elbo_end_training = None
        self.hist_loss = None
        self.hist_output_scale = None
        self.hist_length_scale = None

    @staticmethod
    def set_inducing_points(inducing_points):
        if isinstance(inducing_points, list):
            return torch.Tensor(inducing_points).float()
        elif isinstance(inducing_points, np.ndarray):
            return torch.from_numpy(inducing_points.astype(np.float32))
        elif isinstance(inducing_points, torch.Tensor):
            return inducing_points.float()
        else:
            raise ValueError

    @staticmethod
    def set_h_functions(h):

        if h == "sigmoid" or h == torch.sigmoid:
            h = torch.sigmoid
            h_inv = torch.logit
            h_second = sigmoid_second_derivative
        elif h == "exp" or h == torch.exp:
            h = torch.exp
            h_inv = safe_log
            h_second = torch.exp
        elif h == "identity":
            h = identity
            h_inv = identity
            h_second = lambda x: 0
        else:
            raise ValueError

        return h, h_inv, h_second

    def format_data(self, data):

        n_output_total = len(
            [c for c in data.columns if c.startswith("x")])
        # We assume that there are 2 lotteries with `n_output_per_lot` each
        n_output_per_lot = n_output_total // 2

        x = np.hstack(
            [data[f"x{i}"].values for i in range(n_output_total)])
        p = np.hstack(
            [data[f"p{i}"].values for i in range(n_output_total)])
        y = data.choices.values

        train_x = torch.from_numpy(x).float()
        train_p = torch.from_numpy(p).float()
        train_y = torch.from_numpy(y).float()

        return train_x, train_y, train_p, train_y, \
            n_output_total, n_output_per_lot

    def compute_L_eta_T(self, n_samples, covar):

        L = psd_safe_cholesky(
            covar,
            max_tries=self.cholesky_max_tries)
        eta = torch.randn(covar.shape[0], n_samples)
        L_eta = L @ eta
        return L_eta.T

    def compute_f_no_cor(self, h_inv_m, r_mean, L_eta_T):

        mean = h_inv_m + r_mean
        return self.h(mean + L_eta_T)

    def compute_f_cor_mean_specific(self,
                                    h_inv_m,
                                    r_mean,
                                    L_eta_T):

        mean = h_inv_m + r_mean

        if self.h == identity:
            pass

        elif self.h == torch.exp:
            output_scale = self.r_model.covar_module.outputscale
            can_be_corr = torch.isfinite(mean)
            mean_to_c = mean[can_be_corr]
            mean_to_c -= 0.5 * torch.ones_like(mean_to_c) * output_scale
            mean[can_be_corr] = mean_to_c

        elif self.h == torch.sigmoid:
            output_scale = self.r_model.covar_module.outputscale
            can_be_corr = torch.isfinite(mean)
            mean_to_c = mean[can_be_corr]
            sig_mean = torch.sigmoid(mean_to_c)
            norm_icdf = torch.distributions.Normal(0.0, 1.0).icdf(sig_mean)
            term_covar = torch.sqrt(output_scale + 0.588 ** (-2))
            corr_mean = norm_icdf * term_covar
            mean[can_be_corr] = corr_mean
        else:
            raise ValueError

        out = self.h(mean + L_eta_T)
        return out

    def compute_f_cor_mean_taylor(self,
                                  h_inv_m,
                                  r_mean,
                                  L_eta_T):

        output_scale = self.r_model.covar_module.outputscale

        mean = h_inv_m + r_mean

        base = self.h(mean + L_eta_T)
        corr = - 0.5 * self.h_second(h_inv_m) * output_scale
        return base + corr

    def compute_f(self, h_inv_m, r_mean, L_eta_T):

        kwargs = dict(
            h_inv_m=h_inv_m,
            r_mean=r_mean,
            L_eta_T=L_eta_T)

        if self.mean_correction == 0:
            f = self.compute_f_no_cor(**kwargs)
        elif self.mean_correction == 1:
            f = self.compute_f_cor_mean_specific(**kwargs)
        elif self.mean_correction == 2:
            f = self.compute_f_cor_mean_taylor(**kwargs)
        else:
            raise ValueError

        return f

    def compute_logits(self, f: torch.Tensor):

        n_samples = f.shape[0]
        est_eu = self.train_p * f
        est_eu = est_eu.reshape(n_samples, self.n_output_total, self.n_y)

        est_diff_eu = est_eu[:, self.n_output_per_lot:, :].sum(axis=1) \
            - est_eu[:, :self.n_output_per_lot, :].sum(axis=1)

        logits = self.tau * est_diff_eu
        return logits

    @staticmethod
    def compute_log_prob(logits: torch.Tensor,
                         observations: torch.Tensor):

        # Seems more stable than to use dist.Bernoulli
        p_choice_B = torch.sigmoid(logits)
        p_choice_y = p_choice_B ** observations \
            * (1 - p_choice_B) ** (1 - observations)
        log_prob = torch.log(p_choice_y + np.finfo(float).eps).mean(0)

        # log_prob = \
        #     dist.Bernoulli(logits=probits).log_prob(observations).mean(0)
        return log_prob

    def expected_log_prob(
            self, 
            observations: torch.Tensor, 
            function_dist: gpytorch.distributions.MultivariateNormal,
            n_samples: int = None):

        if n_samples is None:
            n_samples = self.n_samples

        r_mean = function_dist.loc
        r_covar = function_dist.covariance_matrix

        L_eta_T = self.compute_L_eta_T(covar=r_covar,
                                       n_samples=n_samples)

        f = self.compute_f(h_inv_m=self.h_inv_m,
                           r_mean=r_mean,
                           L_eta_T=L_eta_T)

        logits = self.compute_logits(f)
        log_prob = self.compute_log_prob(logits, observations)
        return log_prob

    def train(self, learning_rate=0.05, epochs=1000, seed=123,
              progress_bar=True,
              progress_bar_desc=None):

        # Seed torch
        torch.random.manual_seed(seed)

        # Make the debug easier
        torch.autograd.set_detect_anomaly(True)

        # Switch to 'train' mode
        self.r_model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.r_model.parameters(),
                                     lr=learning_rate)
        # Loss function
        mll = gpytorch.mlls.VariationalELBO(
            likelihood=self,  # Will call the 'expected_log_prob' method
            model=self.r_model,
            num_data=self.train_y.numel())

        self.hist_loss = []
        self.hist_output_scale = []
        self.hist_length_scale = []

        pbar = tqdm(total=epochs, leave=False, desc=progress_bar_desc) \
            if progress_bar else None

        # for name, p in self.r_model.named_parameters():
        #     if p.requires_grad:
        #         print(f"{name}: {p.data.shape}")
        # out:
        # variational_strategy._variational_distribution.variational_mean: torch.Size([50])
        # variational_strategy._variational_distribution.chol_variational_covar: torch.Size([50, 50])
        # covar_module.raw_outputscale: torch.Size([])   # Contains only one item
        # covar_module.base_kernel.raw_lengthscale: torch.Size([1, 1])  # Contains also one item (but different tensor shape)
        for i in range(epochs):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = self.r_model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            # for name, param in self.r_model.named_parameters():
            #     if param.requires_grad:
            #         print(f"gradient {name}", param.grad.norm())
            optimizer.step()

            loss_value = loss.item()
            output_scale = self.r_model.covar_module.outputscale.item()
            length_scale = self.r_model.covar_module.base_kernel.lengthscale.item()
            self.hist_loss.append(loss_value)
            self.hist_output_scale.append(output_scale)
            self.hist_length_scale.append(length_scale)

            if pbar is not None:
                pbar.set_postfix(loss=loss.item())
                pbar.update()

        if pbar is not None:
            pbar.close()

        self.r_model.eval()

        with torch.no_grad():
            # Get predictive output
            output = self.r_model(self.train_x)
            # Calc loss and backprop gradients
            self.elbo_end_training = mll(output, self.train_y)

        return self.hist_loss

    def pred(self, test_x, n_samples=1000):

        # Switch to 'eval' mode
        self.r_model.eval()

        with torch.no_grad():

            r_dist = self.r_model(test_x)  # .sample(torch.Size((n_sample,)))
            r_mean_pred = r_dist.loc
            r_covar_pred = r_dist.covariance_matrix
            m_pred = self.u(test_x, self.theta)
            h_inv_m_pred = self.h_inv(m_pred)
            L_eta_T = self.compute_L_eta_T(covar=r_covar_pred,
                                           n_samples=n_samples)

            f_pred = self.compute_f(h_inv_m=h_inv_m_pred,
                                    r_mean=r_mean_pred,
                                    L_eta_T=L_eta_T)
        return m_pred, f_pred
