import torch
import warnings
import numpy as np
from itertools import product
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F


# Class for ListMLE loss function


class Loss_ListMLE(nn.Module):
    def __init__(self):
        super(Loss_ListMLE, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-10):
        """
        Compute ListMLE loss function for a batch.

        Parameters
        ----------
        y_pred : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        y_true : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        eps : float
            eps used for numerical stability.

        Returns
        -------
        float
            ListMLE loss value.
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_schuffle = y_pred[:, random_indices]
        y_true_schuffle = y_true[:, random_indices]

        y_true_sorted, indices = y_true_schuffle.sort(descending=True, dim=-1)

        preds_sorted_by_true = torch.gather(y_pred_schuffle, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(
            preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        return torch.mean(torch.sum(observation_loss, dim=1))


# Class for ListMLE_corrected loss function


class Loss_ListMLE_corrected(nn.Module):
    def __init__(self):
        super(Loss_ListMLE_corrected, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-10):
        """
        Compute ListMLE_corrected loss function for a batch.

        Parameters
        ----------
        y_pred : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        y_true : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        eps : float
            eps used for numerical stability.

        Returns
        -------
        float
            ListMLE_corrected loss value.
        """

        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values.detach()

        cumsums = torch.cumsum(
            preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        stocks = y_pred.size(1)
        weights = torch.arange(
            stocks, 0, -1, dtype=torch.float32, device=y_pred.device
        ).unsqueeze(0)

        weighted_observation_loss = weights * (
            torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
        )

        return torch.mean(torch.sum(weighted_observation_loss, dim=1))


# Class for ListNET loss function


class Loss_ListNET(nn.Module):
    def __init__(self):
        super(Loss_ListNET, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-10):
        """
        Compute ListNET loss function for a batch.

        Parameters
        ----------
        y_pred : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        y_true : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        eps : float
            eps used for numerical stability.

        Returns
        -------
        float
            ListNET loss value.
        """

        y_pred = y_pred.clone()
        y_true = y_true.clone()

        y_pred = y_pred.float()
        y_true = y_true.float()

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


# Class for ListNET_pairs loss function


class Loss_ListNET_pairs(nn.Module):
    def __init__(self):
        super(Loss_ListNET_pairs, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute ListNET_pairs loss function for a batch.

        Parameters
        ----------
        y_pred : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        y_true : tensor
            2D tensor containing predicted scores of stocks for batch (shape [batch_size * num_stpcks]).

        Returns
        -------
        float
            ListNET_pairs loss value.
        """

        numperiods, num_stocks = y_pred.shape

        pairwise_diff_true = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        pairwise_diff_pred = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

        pairwise_diff_true = pairwise_diff_true.view(numperiods, -1).float()
        pairwise_diff_pred = pairwise_diff_pred.view(numperiods, -1).float()

        probabilities_true = F.softmax(pairwise_diff_true, dim=1)
        probabilities_pred = F.softmax(pairwise_diff_pred, dim=1)

        loss = -torch.sum(
            probabilities_true * torch.log(probabilities_pred), dim=1
        ).mean(dim=0)

        return loss


# Initialization of the Model


def weights_init(m):
    torch.manual_seed(2)

    classname = m.__class__.__name__

    if classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        m.weight.data.normal_(mean=0.0, std=0.05)
        m.bias.data.fill_(0.05)


# Model


class CMLE(nn.Module):
    def __init__(self, n_features):
        super(CMLE, self).__init__()
        self.n_features = n_features
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        result = torch.sigmoid(self.linear4(x))
        result = result.view(result.shape[0], result.shape[1])
        return result


learning_rate = {
    "listnet": 1e-5,
    "listnet_pairs": 1e-5,
    "listmle": 1e-4,
    "listmle_corrected": 1e-4,
}
