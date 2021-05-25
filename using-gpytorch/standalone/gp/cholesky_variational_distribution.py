import torch
from torch import nn
from torch import distributions as dist


class CholeskyVariationalDistribution(nn.Module):
    """
    A variational distribution that is defined to be a multivariate normal distribution
    with a full covariance matrix.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower
    triangle.

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(self, num_inducing_points, mean_init_std=1e-3):

        super().__init__()

        self.num_inducing_points = num_inducing_points
        self.mean_init_std = mean_init_std

        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)

        self.variational_mean = nn.Parameter(mean_init)
        self.chol_variational_covar = nn.Parameter(covar_init)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self):
        mask = torch.ones(self.chol_variational_covar.shape).tril(0)
        chol_variational_covar = self.chol_variational_covar.mul(mask)
        variational_covar = chol_variational_covar@chol_variational_covar.T
        return dist.MultivariateNormal(self.variational_mean, variational_covar)
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.num_inducing_points])

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
        self.chol_variational_covar.data.copy_(prior_dist.covariance_matrix.cholesky())
