import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import scipy
import math

from tqdm.autonotebook import tqdm

import torch
import torch.distributions as dist
from torch.nn import Module

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood

from pymc3.gp.util import plot_gp_dist

sns.set()


class BernoulliLikelihood(_OneDimensionalLikelihood):
    
    r"""
    Close to `~gpytorch.likelihoods.BernoulliLikelihood` but using the inverse-logit function instead of 
    the standard Normal CDF as activation function for the latent process.
    """
    
    def __init__(self):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples, **kwargs):
        raise NotImplementedError

    def log_marginal(self,  *args, **kwargs):
        raise NotImplementedError

    def marginal(self, function_dist, **kwargs):
        raise NotImplementedError

    def expected_log_prob(self, observations, function_dist, n_sample=40, *params, **kwargs):
        
        def log_prob_lambda(function_samples): 
            return dist.Bernoulli(logits=function_samples).log_prob(observations)
        
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob


class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


def true_f(x):
    return np.sin(10*x * np.pi * 0.5) * 10


def main():
    # Seed the random number generators
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Create some toy data
    n = 500
    x = np.sort(np.random.uniform(0, 1, n))
    f = true_f(x)
    y = scipy.stats.bernoulli.rvs(scipy.special.expit(f))
    
    ## Uncomment to show raw data
    # plt.scatter(x, y, alpha=0.5)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.yticks([0, 1])
    # plt.show()

    ## Uncomment to show logits ("f")
    # fig, ax = plt.subplots()
    # x_plot = np.linspace(0, 1, 100)
    # ax.plot(x_plot, true_f(x_plot), alpha=0.5)
    # ax.scatter(x, f, alpha=0.5)
    # plt.show()

    train_x = torch.from_numpy(x.astype(np.float32))
    train_y = torch.from_numpy(y.astype(np.float32))
    
    # Set initial inducing points
    inducing_points = torch.rand(50)

    # Initialize model and likelihood
    model = GPClassificationModel(inducing_points=inducing_points)
    likelihood = BernoulliLikelihood()
    
    # Set number of epochs
    training_iter = 1000

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # num_data refers to the number of training datapoints
    mll = VariationalELBO(likelihood, model, train_y.numel())

    iterator = tqdm(range(training_iter))

    for _ in iterator:

        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()

        iterator.set_postfix(loss=loss.item())
    
    # Show results
    test_x = torch.linspace(0, 1, 101)
    f_preds = model(test_x)

    pred = f_preds.sample(torch.Size((1000,))).numpy()

    fig, ax = plt.subplots()
    plot_gp_dist(ax, pred, test_x)
    ax.plot(test_x, true_f(test_x), alpha=0.5)
    plt.show()
    
    ## Uncomments to show only a few samples
    # fig, ax = plt.subplots()
    # for i in range(10):
    #     ax.plot(test_x, pred[i])
    # plt.show()
    

if __name__ == "__main__":
    main()