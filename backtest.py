import os
import torch
import numpy as np
import pandas as pd

from model import CMLE
from src.ut_test import predict_test, compute_financial_metrics
from src.ut_ranking import (
    compute_ranks_per_batch,
    compute_rank_metrics_per_batch,
    track_accuracy,
    track_accuracy_k,
)


def backtest_ind(
    pond,
    short,
    train_rolling_length,
    test_rolling_length,
    epochs,
    batch_size,
    loss_type,
    ind_run,
    ind_train,
    test_features,
    test_returns,
    nb_features,
    nb_stocks,
    nb_long_max,
    reb_fees,
    positions,
):
    """
    Backtest the strategies for a given window and loss.
    During backtesting, files are saved to track the evolution of ranking and financial metrics over time.

    Parameters
    ----------

    pond : str
        type of portfiolio construction. Should be "weighted" or "unweighted.

    short : str
        type of short position. Should be "bottom" or "average".

    train_rolling_length : int
        length of the training rolling period.

    epochs : int
        number of epochs.

    batch_size : int
        size of the batch.

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected".

    ind_run : int
        id of the run.

    ind_train : int
        id of the window.

    test_features : numpy array
        3D numpy array containing features of various stocks accross time.

    test_returns : numpy array
        2D numpy array containing returns of various stocks accross time.

    nb_features : int
        number of features used.

    nb_stocks : int
        number of stocks used.

    nb_long_max : int
        maximum number of long stocks.

    reb_fees : float
        fees en %.

    positions : np array
        3D numpy array containing 0.


    Returns
    -------


    """

    folder_model_train = f"{os.environ['BASE_PATH']}/train/relu_models/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"
    model_path = folder_model_train + f"/model_train{ind_train}.dat"

    model = CMLE(nb_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    M_predicted_scores = predict_test(model, test_features, batch_size)
    true_ranks_all = compute_ranks_per_batch(test_returns)
    precicted_ranks_all = compute_ranks_per_batch(M_predicted_scores)

    ic, kendalltau, wkendalltau = compute_rank_metrics_per_batch(
        precicted_ranks_all, true_ranks_all
    )
    ic_m = ic.mean()
    kendalltau_m = kendalltau.mean()
    wkendalltau_m = wkendalltau.mean()
    top_acc, bottom_acc = track_accuracy(
        torch.from_numpy(M_predicted_scores).float(),
        torch.from_numpy(test_returns).float(),
    )

    rank_metrics_dico = {
        "ic": [ic_m],
        "kendalltau": [kendalltau_m],
        "wkendalltau": [wkendalltau_m],
        "acc_top10": [top_acc],
        "bot_top10": [bottom_acc],
    }
    df_rank_metrics = pd.DataFrame(rank_metrics_dico)

    group_dico = track_accuracy_k(
        torch.from_numpy(M_predicted_scores).float(),
        torch.from_numpy(test_returns).float(),
    )
    group_dico = {f"acc group{key}": [value] for key, value in group_dico.items()}
    df_acc_group = pd.DataFrame(group_dico)

    df_rank_metrics_final = pd.concat(
        [df_rank_metrics.reset_index(drop=True), df_acc_group.reset_index(drop=True)],
        axis=1,
    )

    L_M_returns_ls = []

    for nb_long in range(1, nb_long_max + 1):
        (
            M_returns_ls,
            M_num_asset,
            M_num_asset_long,
            M_num_asset_short,
            M_weights,
        ) = compute_financial_metrics(
            nb_long, M_predicted_scores, test_returns, pond, short, nb_stocks
        )

        for time in range(len(M_returns_ls)):
            positions[
                nb_long - 1,
                test_rolling_length * ind_train + time + 1,
                M_num_asset_long[time],
            ] += M_weights
            positions[
                nb_long - 1,
                test_rolling_length * ind_train + time + 1,
                M_num_asset_short[time],
            ] -= M_weights

            abs_diff = np.abs(
                positions[nb_long - 1, time + 1] - positions[nb_long - 1, time]
            )
            sum_abs_diff = np.sum(abs_diff)

            sum_abs_weights = np.sum(np.abs(positions[nb_long - 1, time + 1]))
            pct_change = sum_abs_diff / sum_abs_weights

            for i in range(len(M_returns_ls)):
                M_returns_ls[i] -= reb_fees * pct_change

        L_M_returns_ls.append(M_returns_ls)

    df_returns_ind = pd.DataFrame(L_M_returns_ls).T
    columns_returns = [f"ret ls-{i}" for i in range(1, nb_long_max + 1)]
    df_returns_ind.columns = columns_returns

    df_asset_long_ind = pd.DataFrame(M_num_asset_long)
    columns_asset_long = [f"num long-{i}" for i in range(1, nb_long_max + 1)]
    df_asset_long_ind.columns = columns_asset_long

    df_asset_short_ind = pd.DataFrame(M_num_asset_short)
    columns_asset_short = [f"num short-{i}" for i in range(1, nb_long_max + 1)]
    df_asset_short_ind.columns = columns_asset_short

    df_predicted_scores_ind = pd.DataFrame(M_predicted_scores)
    columns_scores = [f"score asset{i}" for i in range(1, nb_long_max * 2 + 1)]
    df_predicted_scores_ind.columns = columns_scores

    df_predicted_scores_ind.index = [
        i for i in range(1, len(df_predicted_scores_ind) + 1)
    ]
    df_returns_ind.index = [i for i in range(1, len(df_returns_ind) + 1)]
    df_asset_long_ind.index = [i for i in range(1, len(df_asset_long_ind) + 1)]
    df_asset_short_ind.index = [i for i in range(1, len(df_asset_short_ind) + 1)]

    return (
        df_predicted_scores_ind,
        df_returns_ind,
        df_asset_long_ind,
        df_asset_short_ind,
        df_rank_metrics_final,
    )


def backtest(
    L_ind_train,
    pond,
    short,
    train_rolling_length,
    test_rolling_length,
    epochs,
    batch_size,
    loss_type,
    ind_run,
    nb_features,
    nb_stocks,
    nb_long_max,
    reb_fees,
    positions,
):
    """
    Backtest the strategies for all windows and a given loss.
    During backtesting, files are saved to track the evolution of ranking and financial metrics over time.

    Parameters
    ----------

    L_ind_train : list
        list containing all the window ids.

    pond : str
        type of portfiolio construction. Should be "weighted" or "unweighted".

    short : str
        type of short position. Should be "bottom" or "average".

    train_rolling_length : int
        length of the training rolling period.

    epochs : int
        number of epochs.

    batch_size : int
        size of the batch.

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected".

    ind_run : int
        id of the run.

    nb_features : int
        number of features used.

    nb_stocks : int
        number of stocks used.

    nb_long_max : int
        maximum number of long stocks.

    reb_fees : float
        fees en %.

    positions : np array
        3D numpy array containing 0.


    Returns
    -------


    """

    (
        L_df_predicted_scores,
        L_df_returns,
        L_df_asset_long,
        L_df_asset_short,
        L_df_rank_metrics_final,
    ) = ([], [], [], [], [])

    for ind_train in range(len(L_ind_train)):
        test_features = np.load(
            f"{os.environ['BASE_PATH']}/data/rolling_300_16/features_test_{ind_train}.npy"
        )
        test_returns = np.load(
            f"{os.environ['BASE_PATH']}/data/rolling_300_16/ranks_test_{ind_train}.npy"
        )

        (
            df_predicted_scores_ind,
            df_returns_ind,
            df_asset_long_ind,
            df_asset_short_ind,
            df_rank_metrics_final,
        ) = backtest_ind(
            pond,
            short,
            train_rolling_length,
            test_rolling_length,
            epochs,
            batch_size,
            loss_type,
            ind_run,
            ind_train,
            test_features,
            test_returns,
            nb_features,
            nb_stocks,
            nb_long_max,
            reb_fees,
            positions,
        )

        L_df_predicted_scores.append(df_predicted_scores_ind)
        L_df_returns.append(df_returns_ind)
        L_df_asset_long.append(df_asset_long_ind)
        L_df_asset_short.append(df_asset_short_ind)
        L_df_rank_metrics_final.append(df_rank_metrics_final)

    df_predicted_scores = pd.concat(L_df_predicted_scores)
    df_returns = pd.concat(L_df_returns)
    df_asset_long = pd.concat(L_df_asset_long)
    df_asset_short = pd.concat(L_df_asset_short)
    df_rank_metrics_final = pd.concat(L_df_rank_metrics_final)

    df_predicted_scores.index = [i for i in range(1, len(df_predicted_scores) + 1)]
    df_returns.index = [i for i in range(1, len(df_returns) + 1)]
    df_asset_long.index = [i for i in range(1, len(df_asset_long) + 1)]
    df_asset_short.index = [i for i in range(1, len(df_asset_short) + 1)]
    df_rank_metrics_final.index = [i for i in range(1, len(df_rank_metrics_final) + 1)]

    folder_test = f"{os.environ['BASE_PATH']}/test/relu_metrics/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)

    df_returns.to_csv(f"{folder_test}" + "/returns.csv")
    df_predicted_scores.to_csv(f"{folder_test}" + "/predicted_scores.csv")
    df_asset_long.to_csv(f"{folder_test}" + "/asset_long.csv")
    df_asset_short.to_csv(f"{folder_test}" + "/asset_short.csv")
    df_rank_metrics_final.to_csv(f"{folder_test}" + "/rank_metrics.csv")

    return

