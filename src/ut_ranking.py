import torch
import numpy as np
import scipy.stats as stats

from scipy.stats import kendalltau
from scipy.stats import weightedtau


def compute_ic(scores, true_ranks):
    """
    Compute IC coefficent.

    Parameters
    ----------
    scores : numpy array
        1D numpy array containing scores of stocks.

    true_ranks : numpy array
        1D numpy array containing ranks of stocks.

    Returns
    -------
    float
       IC coefficent.

    """
    ic, _ = stats.spearmanr(true_ranks, scores)

    return ic


def weighter_kd(i, nb_stocks):

    """
    Compute weight.

    Parameters
    ----------
    i : int
        position

    nb_stocks : int
        number of stocks

    Returns
    -------
    float
        weight.

    """

    if nb_stocks % 2 == 0:
        m = (nb_stocks - 1) / 2
        res = (i - m) ** 2

    else:
        m = nb_stocks // 2
        res = (i - m) ** 2

    return res + 1


def weigher_func(nb_stocks):
    def weigher(i):
        return weighter_kd(i, nb_stocks)

    return weigher


def compute_rank_metrics_per_batch(scores_all, true_ranks_all):
    """
    Compute IC coefficients, kendalltau coefficients and weightedkendalltau coefficients for a given batch.

    Parameters
    ----------
    scores_all : numpy array
        2D numpy array containing scores of stocks.

    true_ranks_all : numpy array
        2D numpy array containing ranks of stocks.

    Returns
    -------

    L_ic : list
        list of IC coefficients.

    L_kendalltau : list
        list of kendalltau coefficients.

    L_wkendalltau : list
        list of weighted kendalltau coefficients.

    """

    nb_stocks = scores_all.shape[1]
    weights = torch.tensor(
        [weighter_kd(i, nb_stocks) for i in range(nb_stocks - 1, -1, -1)]
    )

    nb_t = scores_all.shape[0]

    L_ic = []
    L_kendalltau = []
    L_wkendalltau = []

    for i in range(nb_t):
        scores_all_t = scores_all[i, :]
        true_ranks_t = true_ranks_all[i, :]

        ic = compute_ic(scores_all_t, true_ranks_t)
        L_ic.append(ic)
        kt = kendalltau(scores_all_t, true_ranks_t)[0]
        L_kendalltau.append(kt)
        w_kt = weightedtau(
            scores_all_t, true_ranks_t, rank=True, weigher=weigher_func(nb_stocks)
        )[0]
        L_wkendalltau.append(w_kt)

    L_ic = np.array(L_ic)
    L_kendalltau = np.array(L_kendalltau)
    L_wkendalltau = np.array(L_wkendalltau)

    return L_ic, L_kendalltau, L_wkendalltau


def compute_ranks_per_batch(returns_array):

    """
    Compute ranks of stocks for a given batch.

    Parameters
    ----------
    returns_array : numpy array
        2D numpy array containing returns of stocks.

    Returns
    -------

    rank_array : numpy array
        2D numpy array containing ranks of stocks.

    """
    
    sorted_indices = np.argsort(-returns_array, axis=-1)  # Tri en ordre décroissant
    rank_array = np.empty_like(returns_array, dtype=int)

    for week in range(returns_array.shape[0]):
        week_indices = sorted_indices[week]
        for rank, stock_index in enumerate(week_indices):
            rank_array[week, stock_index] = rank + 1

    return rank_array


# Accuracy tracking functions
## Top/Bottom
def track_accuracy(ypred, ytrue, top_k=10, bottom_k=10):

    """
    Compute accuracy top_k and bottom_k. 

    Parameters
    ----------
    ypred : list
        list of scores of stocks.

    ytrue : list 
        list of returns of stocks.

    top_k : int 
        accuracy top_k.

    bottom_k : int
        accuracy bottom_k.


    Returns
    -------

    top_intersection : float
        value of accuracy top_k.
    
    bottom_intersection : float
        value of accuracy bottom_k.
    
    """

    nweeks = len(ypred)

    bottom_intersection = 0
    top_intersection = 0
    for week in range(nweeks):
        # Sort ypred and ytrue by scores while keeping track of original indices
        _, top_indices_pred = torch.topk(ypred[week], k=top_k)
        _, bottom_indices_pred = torch.topk(ypred[week], k=bottom_k, largest=False)
        _, top_indices_true = torch.topk(ytrue[week], k=top_k)
        _, bottom_indices_true = torch.topk(ytrue[week], k=bottom_k, largest=False)

        # Calculate the intersection of top and bottom indices
        top_intersection += (
            100
            * len(set(top_indices_pred.tolist()) & set(top_indices_true.tolist()))
            / (top_k * nweeks)
        )
        bottom_intersection += (
            100
            * len(set(bottom_indices_pred.tolist()) & set(bottom_indices_true.tolist()))
            / (nweeks * bottom_k)
        )

    return top_intersection, bottom_intersection


##Deciles

def track_accuracy_k(ypred, ytrue, k=10):

    """
    Compute accuracy per group. 

    Parameters
    ----------
    ypred : list
        list of scores of stocks.

    ytrue : list 
        list of returns of stocks.

    Returns
    -------

    top_intersection : dict
        accuracy per group.
    
    """

    nweeks = len(ypred)
    n_elements = ypred.size(1)

    accuracies = {}

    for week in range(nweeks):
        _, indices_pred = torch.topk(ypred[week], k=n_elements)
        _, indices_true = torch.topk(ytrue[week], k=n_elements)

        for group_start in range(0, n_elements, k):
            group_end = min(group_start + k, n_elements)
            if group_start + k > n_elements:
                print("Attention : Last group not full")
            indices_pred_group = indices_pred[group_start:group_end]
            indices_true_group = indices_true[group_start:group_end]
            intersection = len(
                set(indices_pred_group.tolist()) & set(indices_true_group.tolist())
            )
            accuracy = 100 * intersection / (nweeks * (group_end - group_start))
            group_label = f"{group_start + 1}-{group_end}"
            if group_label in accuracies:
                accuracies[group_label] += accuracy
            else:
                accuracies[group_label] = accuracy

    return accuracies
