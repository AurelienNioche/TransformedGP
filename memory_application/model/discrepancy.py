import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torch import nn
import torch.distributions as dist

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.utils.cholesky import psd_safe_cholesky

from typing import Callable, Union, Iterable


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
            x: np.ndarray,
            y: np.ndarray,
            m: Callable,
            theta: Union[int, float, list],
            h: Union[str, Callable] = None,
            learn_inducing_locations: bool = False,
            jitter: Union[int, float] = 1e-07,
            n_samples: int = 40,
            n_inducing_points: int = 50,
            cholesky_max_tries: int = 1000):

        if h is None or h == "sigmoid" or h == torch.sigmoid:
            h = torch.sigmoid
            h_inv = torch.logit

        else:
            raise ValueError

        self.train_x = torch.from_numpy(x.astype(np.float32))
        self.train_y = torch.from_numpy(y.astype(np.float32))

        self.m = m
        self.theta = theta

        self.h = h
        self.h_inv = h_inv

        self.n_samples = n_samples

        self.n_x = self.train_x.size(0)
        self.n_y = self.train_y.size(0)

        self.jitter = jitter

        self.h_inv_m = self.h_inv(self.m(self.train_x, self.theta))

        inducing_delta = np.linspace(x[:, 0].min(), x[:, 0].max(),
                                     n_inducing_points)
        inducing_rep = np.linspace(x[:, 1].min(), x[:, 1].max(),
                                   n_inducing_points)

        inducing_points = np.concatenate(
            (inducing_delta[None, :], inducing_rep[None, :]), axis=0).T
        inducing_points = torch.from_numpy(inducing_points.astype(np.float32))

        self.r_model = GPClassificationModel(
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations)

        self.cholesky_max_tries = cholesky_max_tries

        self.hist_loss = None

    def expected_log_prob(
            self,
            observations: torch.Tensor,
            function_dist: gpytorch.distributions.MultivariateNormal):

        L = psd_safe_cholesky(function_dist.covariance_matrix,
                              max_tries=self.cholesky_max_tries)

        gp_mean = function_dist.loc
        eta = torch.randn(self.n_x, self.n_samples)
        L_eta = L @ eta
        r = gp_mean + L_eta.T

        f = self.h(self.h_inv_m + r)
        log_prob = dist.Bernoulli(probs=f).log_prob(
            observations).mean(0)
        return log_prob

    def train(self, learning_rate=0.05, epochs=1000, seed=123,
              progress_bar=True):

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

        m_pred = self.m(test_x, self.theta)

        h_inv_m = self.h_inv(m_pred)

        f_pred = self.h(h_inv_m + r_pred)

        return m_pred, f_pred
