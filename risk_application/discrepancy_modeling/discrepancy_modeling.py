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
            use_mean_correction: bool = True):

        if h == "sigmoid" or h == torch.sigmoid:
            h = torch.sigmoid
            h_inv = torch.logit
        elif h == "exp" or h == torch.exp:
            h = torch.exp
            h_inv = safe_log
        elif h == "identity":
            h = identity
            h_inv = identity
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

        self.n_samples = n_samples

        self.jitter = jitter
        self.cholesky_max_tries = cholesky_max_tries
        self.use_mean_correction = use_mean_correction

        self.n_x = self.train_x.size(0)
        self.n_y = self.train_y.size(0)

        self.h_inv_m = self.h_inv(u(self.train_x, self.theta))

        self.r_model = GPClassificationModel(
            inducing_points=torch.linspace(0, 1, n_inducing_points),
            learn_inducing_locations=learn_inducing_locations)

        self.hist_loss = None

    def compute_corrected_mean(self, h_inv_m, r):
        mean_x = h_inv_m + r

        if not self.use_mean_correction or self.h == identity:
            print("not using correction")
            return mean_x

        elif self.h == torch.exp:
            mean_x -= 0.5 * torch.ones_like(mean_x) * self.r_model.covar_module.outputscale
            return mean_x

        elif self.h == torch.sigmoid:
            mean_x = \
                torch.distributions.Normal(0.0, 1.0).icdf(torch.sigmoid(mean_x)) \
                * torch.sqrt(self.r_model.covar_module.outputscale + 0.588 ** (-2))
            return mean_x
        else:
            raise ValueError

    def expected_log_prob(
            self, 
            observations: torch.Tensor, 
            function_dist: gpytorch.distributions.MultivariateNormal):

        gp_mean = function_dist.loc
        L = psd_safe_cholesky(function_dist.covariance_matrix,
                              max_tries=self.cholesky_max_tries)
        eta = torch.randn(self.n_x, self.n_samples)
        L_eta = L @ eta
        r = gp_mean + L_eta.T

        if self.use_mean_correction:
            mean = self.h_inv_m + r
        else:
            mean = self.compute_corrected_mean(h_inv_m=self.h_inv_m, r=r)

        f = self.h(mean)

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

        return self.hist_loss

    def pred(self, test_x, n_sample=1000):

        # Switch to 'eval' mode
        self.r_model.eval()

        r_pred = self.r_model(test_x).sample(torch.Size((n_sample,)))

        m_pred = self.u(test_x, self.theta)

        h_inv_m_pred = self.h_inv(m_pred)

        if self.use_mean_correction:
            mean = h_inv_m_pred + r_pred
        else:
            mean = self.compute_corrected_mean(h_inv_m=h_inv_m_pred, r=r_pred)

        f_pred = self.h(mean)

        return m_pred, f_pred
