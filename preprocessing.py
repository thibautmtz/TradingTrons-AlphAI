import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import copy


def handle_nans(train_data, test_data):

    """
    Function to handle NaN values by replacing them with the mean of the respective feature.
   
    Parameters
    ----------
    train_data : numpy array
        3D numpy array containing features of various stocks accross time for the train set.

    test_data : numpy array
        3D numpy array containing returns of various stocks accross time for the test set.

    Returns
    -------
    train_data : numpy array
        3D numpy array containing features of various stocks accross time for the train set after replacing NaN values.

    test_data : numpy array
        3D numpy array containing returns of various stocks accross time for the test set after replacing NaN values.
    """
     
    train_data = np.asarray(train_data, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)
    train_nan_indices = np.isnan(train_data)
    test_nan_indices = np.isnan(test_data)
    train_col_means = np.nanmean(train_data, axis=0)

    # Replace NaNs in test data with mean of corresponding feature from training data,
    test_data[test_nan_indices] = np.take(
        train_col_means, np.where(test_nan_indices)[1]
    )

    # Replace NaNs in train data with mean of corresponding feature
    train_data[train_nan_indices] = np.take(
        train_col_means, np.where(train_nan_indices)[1]
    )

    return train_data, test_data


# Function to perform min-max scaling
def min_max_scale(train_data, test_data=None, scaler=None):

    """
    Function to handle NaN values by replacing them with the mean of the respective feature.
   
    Parameters
    ----------
    train_data : numpy array
        3D numpy array containing features of various stocks accross time for the train set.

    test_data : numpy array
        3D numpy array containing returns of various stocks accross time for the test set.

    Returns
    -------
    train_data_scaled : numpy array
        3D numpy array containing features of various stocks accross time for the train set after scaling.

    scaler : method of MinMaxScaler

    """
     
    if scaler is None:
        scaler = MinMaxScaler()
        # Fit scaler on training data
        scaler.fit(train_data.reshape(-1, train_data.shape[-1]))

    # Transform training data
    train_data_scaled = scaler.transform(
        train_data.reshape(-1, train_data.shape[-1])
    ).reshape(train_data.shape)

    # Transform test data using the same scaler if provided
    if test_data is not None:
        test_data_scaled = scaler.transform(
            test_data.reshape(-1, test_data.shape[-1])
        ).reshape(test_data.shape)
        return train_data_scaled, test_data_scaled, scaler
    else:
        return train_data_scaled, scaler




def save_preprocessed_data(m, returns, train_rolling_length, test_rolling_length):

    """
    Split the dataset into train and test sets after replacing NaN values and scaling. 

    Parameters
    ----------
    m : numpy array
        3D numpy array containing features of various stocks accross time.

    returns : numpy array
        2D numpy array containing returns of various stocks accross time.

    train_rolling_length : int
        length of the rolling training period.

    test_rolling_length : int
        length of the rolling testing period.

    Returns
    -------
    
    """
    
    scaler = None  # Initialize scaler

    for ind, i in enumerate(range(train_rolling_length, len(m), test_rolling_length)):
        train, test = copy.deepcopy(m[i - train_rolling_length : i, :, :]), copy.deepcopy(
            m[i : i + test_rolling_length, :, :]
        )

        # Handle NaNs in train and test data
        train, test = handle_nans(train, test)

        # Preprocess train and test data, and get scaler
        train_scaled, test_scaled, scaler = min_max_scale(train, test, scaler)

        # Save preprocessed data
        if not os.path.exists(
            f"{os.environ['BASE_PATH']}/data" + "/rolling_" + str(train_rolling_length) + "_" + str(test_rolling_length)
        ):
            os.makedirs(
                f"{os.environ['BASE_PATH']}/data" + "/rolling_" + str(train_rolling_length) + "_" + str(test_rolling_length)
            )
        np.save(
            f"{os.environ['BASE_PATH']}/data"
            + "/rolling_"
            + str(train_rolling_length)
            + "_"
            + str(test_rolling_length)
            + "/features_train_"
            + str(ind)
            + ".npy",
            train_scaled,
        )
        np.save(
            f"{os.environ['BASE_PATH']}/data"
            + "/rolling_"
            + str(train_rolling_length)
            + "_"
            + str(test_rolling_length)
            + "/features_test_"
            + str(ind)
            + ".npy",
            test_scaled,
        )
        np.save(
            f"{os.environ['BASE_PATH']}/data"
            + "/rolling_"
            + str(train_rolling_length)
            + "_"
            + str(test_rolling_length)
            + "/ranks_train_"
            + str(ind)
            + ".npy",
            returns[i - train_rolling_length : i],
        )
        np.save(
            f"{os.environ['BASE_PATH']}/data"
            + "/rolling_"
            + str(train_rolling_length)
            + "_"
            + str(test_rolling_length)
            + "/ranks_test_"
            + str(ind)
            + ".npy",
            returns[i : i + test_rolling_length],
        )

    return
    
