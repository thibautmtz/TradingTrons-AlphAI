import numpy as np
import torch
from torch.autograd import Variable


def return_rank(M):
    """
    Compute rank of elements in an array.

    Parameters
    ----------
    M : numpy array

    Returns
    -------
    numpy array
        Rank of elements in an array.

    """

    M = M * -1
    order = M.argsort()

    return order.argsort()


def compute_financial_metrics(k, predicted_scores, returns, pond, short, nb_stocks):
    """
    Compute financials metrics of the strategy ls-k for given predicted_scores and returns.

    Parameters
    ----------
    k : int
        number of long/short stocks

    predicted_scores : numpy array

    returns : numpy array

    pond : str
        type of portfiolio construction. Should be "weighted" or "unweighted.

    short : str
        type of short position. Should be "bottom" or "average".

    nb_stocks : int
        number of stocks used.

    Returns
    -------

    res : numpy array
        time series returns of the strategy.

    num_asset : numpy array
        time series stocks id of the strategy.

    num_asset_long : numpy array
        time series long stocks id of the strategy.

    num_asset_short :numpy array
        time series short stocks id of the strategy.

    weights : numpy array
        weights of stocks for portfolio construction.

    """

    res = []
    num_asset = []
    num_asset_long, num_asset_short = [], []

    for i in range(len(returns)):
        res_t = []
        num_asset_long_t, num_asset_short_t = [], []

        rank = return_rank(predicted_scores[i])

        rank2ind = np.zeros(len(rank), dtype=int)

        for j in range(len(rank)):
            rank2ind[rank[j]] = j

        weights = [0] * k

        s = k * (k + 1) / 2.0

        for j in range(k):
            if pond == "weighted":
                weights[j] = (k - j) / s

            elif pond == "unweighted":
                weights[j] = 1.0 / k

            else:
                raise ValueError(f"{pond} does bot exist")

            num_asset_long_t.append(rank2ind[j])
            num_asset_short_t.append(rank2ind[nb_stocks - 1 - j])

        total_return = 0

        for l in range(k):
            total_return += weights[l] * returns[i][rank2ind[l]]

            if short == "bottom":
                total_return -= weights[l] * returns[i][rank2ind[nb_stocks - 1 - l]]

        if short == "average":
            for h in range(nb_stocks):
                total_return -= 1.0 / (nb_stocks) * returns[i][rank2ind[h]]

        res.append(total_return)
        num_asset_long.append(num_asset_long_t)
        num_asset_short.append(num_asset_short_t)

    return (
        np.array(res),
        np.array(num_asset),
        np.array(num_asset_long),
        np.array(num_asset_short),
        np.array(weights),
    )


def predict_test(model, test_features, batch_size):
    """
    Predictions using the model.

    Parameters
    ----------
    model : object of the CMLE class
        model used.

    test_features : numpy array
        3D numpy array containing features of various stocks accross time.

    batch_size : int
        size of the batch.

    Returns
    -------
    numpy array
        time series predicted scores for stocks over time.

    """

    L = test_features.shape[0]
    N = L // batch_size + 1

    v = np.zeros((N * batch_size, test_features.shape[1], test_features.shape[2]))
    v[:L, :, :] = test_features

    res = []

    for i in range(N):
        batch_x = Variable(
            torch.from_numpy(v[i * batch_size : (i + 1) * batch_size, :, :]).float()
        )
        scores = model(batch_x)
        res.append(np.array(scores.data.cpu()))

    res = np.concatenate(res, axis=0)
    res = res[:L]

    return res
