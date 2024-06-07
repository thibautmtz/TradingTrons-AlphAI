import os
import math as m
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, HTML


def compute_cumret(df_returns):
    """
    Compute cumulative returns of the strategies

    Parameters
    ----------
    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    Returns
    -------
    df
        DataFrame illustrating the time-series cumulative returns of strategies ls_k.

    """

    df_returns.index = [i for i in range(1, len(df_returns) + 1)]
    df_returns_cumsum = df_returns.cumsum()
    df_cumret = df_returns_cumsum + 1
    new_row = pd.DataFrame([[1] * df_cumret.shape[1]], columns=df_cumret.columns)
    df_cumret = pd.concat([new_row, df_cumret], ignore_index=True)
    df_cumret.rename(columns=lambda x: x.replace("ret", "cumret"), inplace=True)

    return df_cumret


def compute_mean_ret(df_returns):
    """
    Compute mean returns of the strategies

    Parameters
    ----------
    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    Returns
    -------
    df
        DataFrame illustrating mean returns of strategies ls_k.

    """

    df_mreturns = pd.DataFrame(df_returns.mean()).T
    df_mreturns.rename(columns=lambda x: x.replace("ret", "mret"), inplace=True)

    return df_mreturns


def compute_volatility(freq, df_returns):
    """
    Compute volatility of the strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    Returns
    -------
    df
        DataFrame illustrating the volatility of strategies ls_k.

    """

    vol = df_returns.std()
    df_vol = pd.DataFrame(vol).T
    df_vol.rename(columns=lambda x: x.replace("ret", "vol"), inplace=True)
    if freq == "daily":
        df_vol = df_vol * np.sqrt(252)
    elif freq == "weekly":
        df_vol = df_vol * np.sqrt(52)

    return df_vol


def compute_excess_return(freq, df_returns, annualized_risk_free_rate):
    """
    Compute excess returns of strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating the time-series excess returns of strategies ls_k.

    """

    if freq == "daily":
        df_returns = df_returns * 252
    elif freq == "weekly":
        df_returns = df_returns * 52
    else:
        raise ValueError("Not adapted freq")
    df_excess_return = df_returns - annualized_risk_free_rate
    df_excess_return.rename(columns=lambda x: x.replace("ret", "ex ret"), inplace=True)

    return df_excess_return


def compute_mean_excess_return(freq, df_returns, annualized_risk_free_rate):
    """
    Compute mean excess returns of strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating mean excess returns of strategies ls_k.

    """

    df_excess_returns = compute_excess_return(
        freq, df_returns, annualized_risk_free_rate
    )
    df_mean_excess_returns = pd.DataFrame(df_excess_returns.mean()).T
    df_mean_excess_returns.rename(columns=lambda x: "m" + "" + x, inplace=True)

    return df_mean_excess_returns


def compute_downside_deviation(freq, df_returns, annualized_risk_free_rate):
    """
    Compute downside_deviation of strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating downside_deviation of strategies ls_k.

    """

    df_excess_returns = compute_excess_return(
        freq, df_returns, annualized_risk_free_rate
    )
    df_excess_returns_neg = df_excess_returns.applymap(lambda x: x if x < 0 else None)
    if freq == "daily":
        df_excess_returns_neg = df_excess_returns_neg / 252
    elif freq == "weekly":
        df_excess_returns_neg = df_excess_returns_neg / 52
    else:
        raise ValueError("Not adapted freq")
    dd = compute_volatility(freq, df_excess_returns_neg)
    dd.rename(columns=lambda x: x.replace("ex vol", "dd"), inplace=True)

    return dd


def compute_sharpe_ratio(freq, df_returns, annualized_risk_free_rate):
    """
    Compute sharpe ratio of strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating sharpe ratio of strategies ls_k.

    """

    df_excess_returns = compute_mean_excess_return(
        freq, df_returns, annualized_risk_free_rate
    )
    df_volatility = compute_volatility(freq, df_returns)
    annual_SR_values = df_excess_returns.values / df_volatility.values
    df_annual_SR = pd.DataFrame(
        annual_SR_values,
        columns=[col.replace("vol", "SR") for col in df_volatility.columns],
    )

    return df_annual_SR


def compute_sortino_ratio(freq, df_returns, annualized_risk_free_rate):
    """
    Compute sortino ratio of strategies

    Parameters
    ----------

    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating sortino ratio of strategies ls_k.

    """

    df_mean_excess_returns = compute_mean_excess_return(
        freq, df_returns, annualized_risk_free_rate
    )
    df_dd = compute_downside_deviation(freq, df_returns, annualized_risk_free_rate)
    annual_sortino_values = df_mean_excess_returns.values / df_dd.values
    df_annual_sortino_ratio = pd.DataFrame(
        annual_sortino_values,
        columns=[col.replace("dd", "sortino") for col in df_dd.columns],
    )

    return df_annual_sortino_ratio


def compute_max_drawdown(df_returns):
    """
    Compute maximum drawdown of strategies

    Parameters
    ----------
    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    Returns
    -------
    df
        DataFrame illustrating maximum drawdown of strategies ls_k.

    """

    df_cum_ret = compute_cumret(df_returns)
    peak = df_cum_ret.cummax()
    drawdown = (df_cum_ret - peak) / peak
    drawdown_values = drawdown.min()
    df_drawdown = pd.DataFrame(drawdown_values).T
    df_drawdown.columns = [col.replace("cumret", "MDD") for col in df_cum_ret.columns]

    return df_drawdown


def compute_calmar_ratio(freq, df_returns, annualized_risk_free_rate):
    """
    Compute calmar ratio of strategies

    Parameters
    ----------
    freq : str
        Frequency of the data: should be "daily" or "weekly".

    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------
    df
        DataFrame illustrating calmar ratio of strategies ls_k.

    """

    df_excess_returns = compute_excess_return(
        freq, df_returns, annualized_risk_free_rate
    )
    df_mdd = compute_max_drawdown(df_returns)
    calmar_values = df_excess_returns.values / abs(df_mdd.values)
    df_calmar = pd.DataFrame(
        calmar_values, columns=[col.replace("MDD", "calmar") for col in df_mdd.columns]
    )

    return df_calmar


def value_at_risk(returns, alpha):
    """
    Compute VaR

    Parameters
    ----------
    returns : pandas series
        returns of strategy

    alpha : float
        threshold

    Returns
    -------
    float
        VaR

    """

    return np.percentile(returns, 100 * alpha)


def compute_value_at_risk(df_returns, alpha=0.05):
    """
    Compute VaR of strategies

    Parameters
    ----------
    df_returns : df
        DataFrame illustrating the time-series returns of strategies ls_k.

    Returns
    -------
    df
        DataFrame illustrating VaR of strategies ls_k.

    """

    VaR = df_returns.apply(value_at_risk, alpha=alpha)
    df_vAR = pd.DataFrame(VaR).T * 100
    df_vAR.columns = [col.replace("ret", "VaR") for col in df_returns.columns]

    return df_vAR


def plot_cumret_ls(df_returns, ls):
    """
    Plot cumulative returns of strategy

    Parameters
    ----------
    df_returns : df
        DataFrame illustrating the time-series returns of strategy ls_k.

    ls : int
        strategy k (k long and k short)

    Returns
    -------

    """

    df_cumret = compute_cumret(df_returns)
    plt.figure(figsize=(10, 6))
    plt.plot(
        [i for i in range(len(df_cumret[f"cumret ls-{ls}"]))],
        df_cumret[f"cumret ls-{ls}"].values,
    )
    plt.title("Cumulative returns of the strategy")
    plt.xlabel("Weeks")
    plt.ylabel("Cumulative returns")

    return


def plot_resume_cumret_ls(L_loss, ls, train_rolling_length, epochs, ind_run):
    """
    Plot cumulative returns of strategy ls for loss function in the list L_loss

    Parameters
    ----------
    L_loss : list
       list containing at least one of the following : "listnet", "listnet_pairs", "listmle", "listmle_corrected".

    ls : int
        strategy k (k long and k short)

    train_rolling_length : int
        length of the training period

    epochs : int
        number of epochs

    ind_run : int
        if of run

    Returns
    -------

    """

    plt.figure(figsize=(10, 6))
    for loss_type in L_loss:
        df_returns = pd.read_csv(
            f"{os.environ['BASE_PATH']}/test/relu_metrics/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"
            + "/returns.csv",
            index_col=0,
        )
        df_cumret = compute_cumret(df_returns)
        plt.plot(
            [i for i in range(len(df_cumret[f"cumret ls-{ls}"]))],
            df_cumret[f"cumret ls-{ls}"].values,
        )
    plt.title("Cumulative returns of the strategy")
    plt.xlabel("Weeks")
    plt.ylabel("Cumulative returns")
    plt.legend(L_loss)

    return


def get_rank_metrics_test(loss_type, train_rolling_length, epochs, ind_run):

    """
    Get rank metrics of the test set for a given loss.

    Parameters
    ----------

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected"

    train_rolling_length : int
        length of the training period

    epochs : int
        number of epochs

    ind_run : int
        if of run

    Returns
    -------

    df_rank_metrics : df
        DataFrame of ranking metrics for a given loss.
    """

    df_rank_metrics = pd.read_csv(
        f"{os.environ['BASE_PATH']}/test/relu_metrics/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"
        + "/rank_metrics.csv",
        index_col=0,
    )
    df_rank_metrics.insert(0, "ls", range(1, len(df_rank_metrics) + 1))

    return df_rank_metrics


def get_resume_rank_metrics_test(train_rolling_length, epochs, ind_run):

    """
    Get rank metrics of the test set for all losses.

    Parameters
    ----------

    train_rolling_length : int
        length of the training period

    epochs : int
        number of epochs

    ind_run : int
        if of run

    Returns
    -------

    df_rank_metrics : df
        DataFrame of ranking metrics for all losses 
    """
        
    df1 = get_rank_metrics_test("listnet_pairs", train_rolling_length, epochs, ind_run)
    df2 = get_rank_metrics_test("listnet", train_rolling_length, epochs, ind_run)
    df3 = get_rank_metrics_test(
        "listmle_corrected", train_rolling_length, epochs, ind_run
    )
    df4 = get_rank_metrics_test("listmle", train_rolling_length, epochs, ind_run)
    df1 = df1.mean().round(3)
    df2 = df2.mean().round(3)
    df3 = df3.mean().round(3)
    df4 = df4.mean().round(3)
    df_rank_metrics = pd.concat([df1, df2, df3, df4], axis=1)[1:4]
    df_rank_metrics.columns = [
        "ListNet Classic",
        "ListNet Fold",
        "ListMLE Classic",
        "ListMLE Weighted",
    ]

    return df_rank_metrics


def get_financial_metrics_test(
    loss_type, train_rolling_length, epochs, ind_run, freq, annualized_risk_free_rate
):
    
    """
    Get financial metrics of the test set for a given loss.

    Parameters
    ----------

    loss_type : str
        loss function "listnet" or "listnet_pairs" or "listmle" or "listmle_corrected"

    train_rolling_length : int
        length of the training period

    epochs : int
        number of epochs

    ind_run : int
        if of run

    freq : str
        Frequency of the data: should be "daily" or "weekly".

    annualized_risk_free_rate : float

        Annualized risk free rate.
        
    Returns
    -------

    df_metrics : df
        DataFrame of financial metrics for a given loss.
    """

    df_returns = pd.read_csv(
        f"{os.environ['BASE_PATH']}/test/relu_metrics/train_rolling_length{train_rolling_length}/epochs{epochs}/loss_{loss_type}/ind_run{ind_run}"
        + "/returns.csv",
        index_col=0,
    )
    vol = compute_volatility(freq, df_returns).values[0]
    mex = compute_mean_excess_return(
        freq, df_returns, annualized_risk_free_rate
    ).values[0]
    SR = compute_sharpe_ratio(freq, df_returns, annualized_risk_free_rate).values[0]
    sortino = compute_sortino_ratio(freq, df_returns, annualized_risk_free_rate).values[
        0
    ]
    MDD = compute_max_drawdown(df_returns).values[0]
    VaR = compute_value_at_risk(df_returns).values[0]
    df_metrics = pd.DataFrame(
        {
            "ex ret": mex,
            "vol": vol,
            "sharpe": SR,
            "MDD": MDD,
            "sortino": sortino,
            "VaR": VaR,
        }
    )

    df_metrics.insert(0, "ls", range(1, len(df_metrics) + 1))

    return df_metrics


def get_resume_financial_metrics_test(
    train_rolling_length, epochs, ind_run, freq, annualized_risk_free_rate
):
    
    """
    Get rank financial metrics of the test set for all losses.

    Parameters
    ----------

    train_rolling_length : int
        length of the training period

    epochs : int
        number of epochs

    ind_run : int
        if of run

    freq : str
        Frequency of the data: should be "daily" or "weekly".

    annualized_risk_free_rate : float

        Annualized risk free rate.

    Returns
    -------

    df_financial_metrics : df
        DataFrame of ranking metrics for all losses 
    """

    df1 = get_financial_metrics_test(
        "listnet_pairs",
        train_rolling_length,
        epochs,
        ind_run,
        freq,
        annualized_risk_free_rate,
    )
    df2 = get_financial_metrics_test(
        "listnet",
        train_rolling_length,
        epochs,
        ind_run,
        freq,
        annualized_risk_free_rate,
    )
    df3 = get_financial_metrics_test(
        "listmle_corrected",
        train_rolling_length,
        epochs,
        ind_run,
        freq,
        annualized_risk_free_rate,
    )
    df4 = get_financial_metrics_test(
        "listmle",
        train_rolling_length,
        epochs,
        ind_run,
        freq,
        annualized_risk_free_rate,
    )
    df1 = pd.DataFrame(df1.loc[df1["sharpe"].idxmax()].round(2)[1:])
    df2 = pd.DataFrame(df2.loc[df2["sharpe"].idxmax()].round(2)[1:])
    df3 = pd.DataFrame(df3.loc[df3["sharpe"].idxmax()].round(2)[1:])
    df4 = pd.DataFrame(df4.loc[df4["sharpe"].idxmax()].round(2)[1:])
    df_financial_metrics = pd.concat([df1, df2, df3, df4], axis=1)
    df_financial_metrics.columns = [
        "ListNet Classic",
        "ListNet Fold",
        "ListMLE Classic",
        "ListMLE Weighted",
    ]

    return df_financial_metrics
