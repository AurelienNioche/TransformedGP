import math

import numpy as np

import torch
from torch import distributions as dist


class GaussHermiteQuadrature1D:

    def __init__(self, num_locs=None):
        if num_locs is None:
            """
            The number of samples to draw from a latent GP when computing a likelihood
            This is used in variational inference and training

            (Default: 20)
            """
            num_locs = 20

        self.num_locs = num_locs

        locations, weights = self._locs_and_weights(num_locs)

        self.locations = locations
        self.weights = weights

    def _apply(self, fn):
        self.locations = fn(self.locations)
        self.weights = fn(self.weights)
        return super(GaussHermiteQuadrature1D, self)._apply(fn)

    def _locs_and_weights(self, num_locs):
        """
        Get locations and weights for Gauss-Hermite quadrature. Note that this is **not** intended to be used
        externally, because it directly creates tensors with no knowledge of a device or dtype to cast to.

        Instead, create a GaussHermiteQuadrature1D object and get the locations and weights from buffers.
        """
        locations, weights = np.polynomial.hermite.hermgauss(num_locs)
        locations = torch.Tensor(locations)
        weights = torch.Tensor(weights)
        return locations, weights

    def forward(self, func, gaussian_dists):
        """
        Runs Gauss-Hermite quadrature on the callable func, integrating against the Gaussian distributions specified
        by gaussian_dists.

        Args:
            - func (callable): Function to integrate
            - gaussian_dists (Distribution): Either a MultivariateNormal whose covariance is assumed to be diagonal
                or a :obj:`torch.distributions.Normal`.
        Returns:
            - Result of integrating func against each univariate Gaussian in gaussian_dists.
        """
        means = gaussian_dists.mean
        variances = gaussian_dists.variance

        locations = self._pad_with_singletons(self.locations,
                                              num_singletons_before=0,
                                              num_singletons_after=means.dim())

        shifted_locs = torch.sqrt(2.0 * variances) * locations + means
        log_probs = func(shifted_locs)
        weights = self._pad_with_singletons(self.weights,
                                            num_singletons_before=0,
                                            num_singletons_after=log_probs.dim() - 1)

        res = (1 / math.sqrt(math.pi)) * (log_probs * weights)
        res = res.sum(tuple(range(self.locations.dim())))

        return res

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @staticmethod
    def _pad_with_singletons(obj, num_singletons_before=0,
                             num_singletons_after=0):
        """
        Pad obj with singleton dimensions on the left and right

        Example:
            >>> x = torch.randn(10, 5)
            >>> _pad_width_singletons(x, 2, 3).shape
            >>> # [1, 1, 10, 5, 1, 1, 1]
        """
        new_shape = [1] * num_singletons_before + list(obj.shape) + [
            1] * num_singletons_after
        return obj.view(*new_shape)


class BernoulliLikelihood:
    r"""
    A specific case of :obj:`~gpytorch.likelihoods.Likelihood` when the GP represents a one-dimensional
    output. (I.e. for a specific :math:`\mathbf x`, :math:`f(\mathbf x) \in \mathbb{R}`.)

    Inheriting from this likelihood reduces the variance when computing approximate GP objective functions
    by using 1D Gauss-Hermite quadrature.
    """

    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF :math:`\Phi(x)`). Given the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood compactly as:
    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi(yf)
        \end{equation*}

    """

    def __init__(self):
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples, **kwargs):
        raise NotImplementedError

    def log_marginal(self, *args, **kwargs):
        raise NotImplementedError

    def marginal(self, function_dist, **kwargs):
        raise NotImplementedError

    def expected_log_prob(self, observations, function_dist):
        def log_prob_lambda(function_samples):
            # function_samples.shape  20 x Nobs
            return dist.Bernoulli(logits=function_samples).log_prob(
                observations)

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob
