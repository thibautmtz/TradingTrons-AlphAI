import os
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from model import (
    CMLE,
    Loss_ListMLE,
    Loss_ListMLE_corrected,
    Loss_ListNET,
    Loss_ListNET_pairs,
    learning_rate,
)

from src.ut_train import random_batch
from src.ut_ranking import (
    compute_ranks_per_batch,
    compute_rank_metrics_per_batch,
    track_accuracy,
    track_accuracy_k,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Rolling Training")

Losses = {
    "listmle": Loss_ListMLE,
    "listmle_corrected": Loss_ListMLE_corrected,
    "listnet": Loss_ListNET,
    "listnet_pairs": Loss_ListNET_pairs,
}


def train_model(
    ind_train,
    train_features,
    train_returns,
    train_rolling_length,
    model,
    loss_type,
    epochs,
    batch_size,
    ind_run,
):
    """
    Train a model for a given window and loss.
    During model training, files are saved to track the evolution of ranking metrics and loss functions over epochs.

    Parameters
    ----------
    ind_train : int
        id of the window

    train_features : numpy array
        3D numpy array containing features of various stocks accross time

    train_returns : numpy array
        2D numpy array containing returns of various stocks accross time

    train_rolling_length : int
        length of the training rolling period

    model : object of the CMLE class
        model used

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected"

    epochs : int
        number of epochs

    batch_size : int
        size of the batch

    ind_run : int
        if of the run

    Returns
    -------
    dict
        dictionary which contains all the optimized parameters of the model

    """

    nb_stocks = train_features.shape[1]

    loss = Losses[loss_type]()
    opt = optim.Adam(model.parameters(), lr=learning_rate[loss_type])

    L_loss = []
    torch.set_grad_enabled(True)

    L_columns = (
        [{"ic": []}, {"kendalltau": []}, {"wkendalltau": []}]
        + [{"acc_bot10": []}, {"acc_top10": []}]
        + [{"scores_min": []}, {"scores_max": []}]
    )

    df_metrics = pd.DataFrame(L_columns)
    df_loss = pd.DataFrame([])

    train_features_tensor = Variable(torch.from_numpy(train_features).float())
    true_ranks_all = compute_ranks_per_batch(train_returns)

    np.random.seed(2)
    ind_batch_epochs = np.random.randint(0, len(train_features), batch_size * epochs)

    for ind_epoch, itr in enumerate(range(epochs)):
        batch_x, batch_y = random_batch(
            train_features,
            train_returns,
            ind_epoch,
            ind_batch_epochs,
            batch_size,
            nb_stocks,
        )

        batch_x = Variable(torch.from_numpy(batch_x).float())
        batch_y = Variable(torch.from_numpy(batch_y).float())

        model.train()
        scores = model(batch_x)

        if loss_type not in Losses:
            raise ValueError("Change loss")

        l = loss(scores, batch_y)

        opt.zero_grad()
        l.backward()
        opt.step()

        L_loss.append(l.mean().item())

        scores_all = model(train_features_tensor)
        scores_all = scores_all.detach().numpy()

        score_min = scores_all.min()
        score_max = scores_all.max()

        precicted_ranks_all = compute_ranks_per_batch(scores_all)

        ic, kendalltau, wkendalltau = compute_rank_metrics_per_batch(
            precicted_ranks_all, true_ranks_all
        )

        ic_m = ic.mean()
        kendalltau_m = kendalltau.mean()
        wkendalltau_m = wkendalltau.mean()

        top_acc, bottom_acc = track_accuracy(
            torch.from_numpy(scores_all).float(),
            torch.from_numpy(train_returns).float(),
        )

        L_metrics = (
            [ic_m]
            + [kendalltau_m]
            + [wkendalltau_m]
            + [top_acc]
            + [bottom_acc]
            + [score_min]
            + [score_max]
        )
        df_metrics.loc[itr] = L_metrics

    group_dico = track_accuracy_k(
        torch.from_numpy(scores_all).float(), torch.from_numpy(train_returns).float()
    )
    group_dico = {key: [value] for key, value in group_dico.items()}
    df_acc_group = pd.DataFrame(group_dico)

    df_loss[f"loss_{loss_type}"] = L_loss

    folder_model_train = f"{os.environ['BASE_PATH']}/train/relu_models/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"

    # folder_ranks_metrics_train_epochs = f"{os.environ['BASE_PATH']}/train/relu_metrics/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}/epochs"

    if not os.path.exists(folder_model_train):
        os.makedirs(folder_model_train)

    # if not os.path.exists(folder_ranks_metrics_train_epochs):
    #   os.makedirs(folder_ranks_metrics_train_epochs)

    torch.save(
        model.state_dict(), folder_model_train + f"/model_train{ind_train}" + ".dat"
    )

    # df_metrics.to_csv(folder_ranks_metrics_train_epochs+ f"/rank_metrics_epochs_train{ind_train}"+ ".csv")
    # df_loss.to_csv(folder_ranks_metrics_train_epochs + f"/loss_epochs_train{ind_train}" + ".csv")
    # df_acc_group.to_csv(folder_ranks_metrics_train_epochs + f"/group_acc_train{ind_train}" + ".csv")

    params = model.state_dict()

    return params


def train_model_loss(
    loss_type,
    L_ind_train,
    train_rolling_length,
    test_rolling_length,
    epochs,
    batch_size,
    ind_run,
):
    """
    Train a model for all windows and a given loss.

    Parameters
    ----------

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected"

    L_ind_train : list
        list containing all the window ids

    train_rolling_length : int
        length of the rolling training period

    test_rolling_length : int
        length of the rolling testing period

    epochs : int
        number of epochs

    batch_size : int
        size of the batch

    ind_run : int
        id of the run

    Returns
    -------

    """

    for ind_train in L_ind_train:

        train_features = np.load(
            f"{os.environ['BASE_PATH']}/data/rolling_{train_rolling_length}_{test_rolling_length}/features_train_{ind_train}.npy"
        )
        train_returns = np.load(
            f"{os.environ['BASE_PATH']}/data/rolling_{train_rolling_length}_{test_rolling_length}/ranks_train_{ind_train}.npy"
        )

        model = CMLE(train_features.shape[2])

        train_model(
            ind_train,
            train_features,
            train_returns,
            train_rolling_length,
            model,
            loss_type,
            epochs,
            batch_size,
            ind_run,
        )

    return

