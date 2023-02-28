import torch
import torch.nn as nn
import torch.nn.functional as F


class VicregLoss(nn.Module):
    def __init__(self, batch_size, num_features=8192, sim_coeff=25, std_coeff=25, cov_coeff=1):
        super(VicregLoss, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, mode = "train"):

        repr_loss = nn.functional.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        if mode == "train":
            return loss
        else:
            return self.sim_coeff * repr_loss, self.std_coeff * std_loss,  self.cov_coeff * cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
