import numpy as np
from typing import Callable, Union

import torch
import torch.distributions as dist

import gpytorch
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


class DiscrepancyModelCPC(DiscrepancyModel):

    def __init__(
            self, h: Union[str, Callable], *args, **kwargs):

        assert h in ("sigmoid", torch.sigmoid), \
            "Actual code version assumes h: sigmoid"

        super().__init__(h=h, *args, **kwargs)

    @staticmethod
    def format_data(data):

        n_output_total = len(
            [c for c in data.columns if c.startswith("x")])
        # We assume that there are 2 lotteries with `n_output_per_lot` each
        n_output_per_lot = n_output_total // 2

        x = np.hstack(
            [data[f"x{i}"].values for i in range(n_output_total)])
        p = np.hstack(
            [data[f"p{i}"].values for i in range(n_output_total)])
        y = data.choices.values

        train_x_unique, train_x_init_order = np.unique(x, return_inverse=True)
        # Remove 0 and 1 as
        # for numerical reasons due to the mean correction,
        # we need to 'force' the assumption that
        # f(0) = 0 and f(1) = 1
        assert train_x_unique[0] == 0 \
               and train_x_unique[-1] == 1, "Code needs to be revised"
        train_x = torch.from_numpy(
            train_x_unique[1:-1]).float()

        train_p = torch.from_numpy(p).float()
        train_y = torch.from_numpy(y).float()

        return train_x, train_y, train_p, train_y, \
            train_x_init_order, \
            n_output_total, n_output_per_lot

    def format_f(self, f: torch.Tensor):

        # For numerical reasons due to the mean correction,
        # we need to 'force' the assumption that
        # f(0) = 0 and f(1) = 1
        _f = torch.zeros((f.shape[0], f.shape[1]+2))
        _f[:, 1:-1] = f
        _f[:, -1] = 1.0

        return _f[:, self.train_x_init_order]

