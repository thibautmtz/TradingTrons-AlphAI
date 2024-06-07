import numpy as np


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


def random_batch(x, y, ind_epoch, ind_batch_epochs, batch_size, nb_stocks):
    """
    Generate random batches.

    Parameters
    ----------
    x : numpy array
        3D numpy array containing features of various stocks accross time

    y : numpy array
        2D numpy array containing returns of various stocks accross time

    ind_epoch : int
        id of the epoch.

    ind_batch_epochs : int
        id of the batch within the epoch.

    batch_size : int
        size of the batch

    nb_stocks : int
        number of stocks used.

    Returns
    -------
    x_sorted : numpy array

        3D numpy array containing features of various stocks accross time sorted by returns

    y_sorted : numpy array

        2D numpy array containing returns of various stocks accross time sorted by returns

    """

    ind = ind_batch_epochs[ind_epoch * batch_size : (ind_epoch + 1) * batch_size]

    batch_x, batch_y = x[ind], y[ind]

    x_sorted = np.zeros(batch_x.shape)
    y_sorted = np.zeros(batch_y.shape)

    for i in range(len(batch_x)):
        rank_temp = return_rank(batch_y[i])
        rank2ind = np.zeros(nb_stocks, dtype=int)

        for j in range(len(rank_temp)):
            rank2ind[rank_temp[j]] = int(j)

        for j in range(len(rank_temp)):
            x_sorted[i, rank_temp[j], :] = batch_x[i][rank2ind[rank_temp[j]]]
            y_sorted[i, rank_temp[j]] = batch_y[i][rank2ind[rank_temp[j]]]

    return x_sorted, y_sorted
