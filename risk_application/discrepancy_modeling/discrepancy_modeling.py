import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
from typing import Callable, Union

import torch
import torch.distributions as dist

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky


class GPClassificationModel(ApproximateGP):

    def __init__(self, inducing_points, learn_inducing_locations):
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


def sigmoid_second_derivative(x):
    e_x = torch.exp(x)
    return - ((e_x - 1) * e_x) / (e_x + 1)**3


class DiscrepancyModel:

    def __init__(
            self,
            data: pd.DataFrame,
            u: Callable,
            theta: Union[int, float],
            h: Union[str, Callable],
            tau: Union[int, float],
            learn_inducing_locations: bool = False,
            jitter: Union[int, float] = 1e-07,
            n_samples: int = 40,
            n_inducing_points: int = 50,
            cholesky_max_tries: int = 1000,
            mean_correction: int = 0):

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

        self.n_output_total = len(
            [c for c in data.columns if c.startswith("x")])

        x = np.hstack(
            [data[f"x{i}"].values for i in range(self.n_output_total)])
        p = np.hstack(
            [data[f"p{i}"].values for i in range(self.n_output_total)])
        y = data.choices.values

        x_order = np.argsort(x)
        x_sorted = x[x_order]
        p_sorted = p[x_order]

        # We assume that there are 2 lotteries with `n_output_per_lot` each
        self.n_output_per_lot = self.n_output_total // 2

        self.init_order = np.argsort(x_order)

        self.train_x = torch.from_numpy(x_sorted.astype(np.float32))
        self.train_p = torch.from_numpy(p_sorted.astype(np.float32))
        self.train_y = torch.from_numpy(y.astype(np.float32))

        self.u = u
        self.theta = theta
        self.tau = tau

        self.h = h
        self.h_inv = h_inv
        self.h_second = h_second

        self.n_samples = n_samples

        self.jitter = jitter
        self.cholesky_max_tries = cholesky_max_tries
        self.mean_correction = mean_correction

        self.n_x = self.train_x.size(0)
        self.n_y = self.train_y.size(0)

        self.h_inv_m = self.h_inv(u(self.train_x, self.theta))

        self.r_model = GPClassificationModel(
            inducing_points=torch.linspace(0, 1, n_inducing_points),
            learn_inducing_locations=learn_inducing_locations)

        self.elbo_end_training = None

        self.hist_loss = None

    def compute_L_eta_T(self, n_samples, covar):
        L = psd_safe_cholesky(
            covar,
            max_tries=self.cholesky_max_tries)
        eta = torch.randn(covar.shape[0], n_samples)
        L_eta = L @ eta
        return L_eta.T

    def compute_f_no_cor(self, h_inv_m, r_mean, r_covar, n_samples):

        mean = h_inv_m + r_mean
        L_eta_T = self.compute_L_eta_T(covar=r_covar,
                                       n_samples=n_samples)
        return self.h(mean + L_eta_T)

    def compute_f_cor_mean_specific(self,
                                     h_inv_m,
                                     r_mean, r_covar, n_samples):

        mean = h_inv_m + r_mean

        if self.h == identity:
            pass

        elif self.h == torch.exp:
            mean -= 0.5 * torch.ones_like(mean) * self.r_model.covar_module.outputscale

        elif self.h == torch.sigmoid:
            output_scale = self.r_model.covar_module.outputscale
            sig_mean = torch.sigmoid(mean)
            mean = \
                torch.distributions.Normal(0.0, 1.0).icdf(sig_mean) \
                * torch.sqrt(output_scale + 0.588 ** (-2))
        else:
            raise ValueError

        L_eta_T = self.compute_L_eta_T(covar=r_covar,
                                       n_samples=n_samples)
        return self.h(mean + L_eta_T)

    def compute_f_cor_mean_taylor(self,
                                      h_inv_m,
                                      r_mean, r_covar, n_samples):

        output_scale = self.r_model.covar_module.outputscale

        L_eta_T = self.compute_L_eta_T(covar=r_covar,
                                       n_samples=n_samples)

        r = r_mean + L_eta_T
        return self.h(h_inv_m + r) \
            - 0.5 * self.h_second(h_inv_m) * output_scale

    def compute_f(self, h_inv_m, r_mean, r_covar, n_samples):

        kwargs = dict(
            h_inv_m=h_inv_m,
            r_mean=r_mean,
            r_covar=r_covar,
            n_samples=n_samples)

        if self.mean_correction == 0:
            f = self.compute_f_no_cor(**kwargs)
        elif self.mean_correction == 1:
            f = self.compute_f_cor_mean_specific(**kwargs)
        elif self.mean_correction == 2:
            f = self.compute_f_cor_mean_taylor(**kwargs)
        else:
            raise ValueError

        return f

    def expected_log_prob(
            self, 
            observations: torch.Tensor, 
            function_dist: gpytorch.distributions.MultivariateNormal):

        r_mean = function_dist.loc
        r_covar = function_dist.covariance_matrix

        f = self.compute_f(h_inv_m=self.h_inv_m,
                           r_mean=r_mean,
                           r_covar=r_covar,
                           n_samples=self.n_samples)

        est_eu_sorted = self.train_p * f
        est_eu = est_eu_sorted[:, self.init_order]

        est_eu = est_eu.reshape(self.n_samples, self.n_output_total, self.n_y)

        est_diff_eu = est_eu[:, self.n_output_per_lot:, :].sum(
            axis=1) - est_eu[:, :self.n_output_per_lot, :].sum(axis=1)

        log_prob = dist.Bernoulli(logits=self.tau * est_diff_eu).log_prob(
            observations).mean(0)
        return log_prob

    def train(self, learning_rate=0.05, epochs=1000, seed=123,
              progress_bar=True):

        # Seed torch
        torch.random.manual_seed(seed)

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

        pbar = tqdm(total=epochs, leave=False) if progress_bar else None

        for i in range(epochs):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = self.r_model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            self.hist_loss.append(loss.item())

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

    def pred(self, test_x, n_sample=1000):

        # Switch to 'eval' mode
        self.r_model.eval()

        with torch.no_grad():

            r_dist = self.r_model(test_x)  # .sample(torch.Size((n_sample,)))
            r_mean_pred = r_dist.loc
            r_covar_pred = r_dist.covariance_matrix
            m_pred = self.u(test_x, self.theta)
            h_inv_m_pred = self.h_inv(m_pred)

            f_pred = self.compute_f(h_inv_m=h_inv_m_pred,
                                    r_mean=r_mean_pred,
                                    r_covar=r_covar_pred,
                                    n_samples=n_sample)

        return m_pred, f_pred
