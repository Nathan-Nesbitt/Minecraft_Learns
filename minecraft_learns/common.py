"""
    Add some common methods for Machine Learning
"""

from numpy import power, sqrt, log
from numpy import sum as np_sum

from pandas import concat, DataFrame

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


def euclidean_distance(a, b):
    """
    find the euclidean distance between a and b
    ---
    @param a: numpy 2D array representing n observations of m predictors
    @param b: numpy array representing one observation with m predictors
    ---
    Formula for Euclidean Distance:
        dist = sqrt(sum((a[1]+b[1])^2 + ... + (a[n]+b[n])^2))
    """
    return sqrt(np_sum(power(a - b, 2), axis=1))


def interact(data, interaction_cols):
    """
    add interactions to data for the columns in self.columns
    ---
    @param data: a 2D data matrix of n observations and m predictors
    @param interaction_cols: list of columns to interact
    """
    # for each pair of interaction columns, create a new product column
    for column_1 in interaction_cols:
        for column_2 in interaction_cols:
            if column_1 != column_2:
                column_name = "" + column_1 + "*" + column_2
                data[column_name] = data[column_1] * data[column_2]
    return data


def one_hot_encode(data, encode_cols):
    """
    one hot encode the dataframe data
    ---
    @param data: a 2D data matrix of n observations and m predictors
    @param encode_cols: list of columns to encode
    """
    # encode the columns
    encoded = data[encode_cols]
    encoded = OneHotEncoder(sparse=False).fit_transform(encoded)
    encoded = DataFrame(encoded, index=data.index)
    # remove the duplicates and concatenate
    return concat([data.drop(encode_cols, axis=1), encoded], axis=1)


def pca(data, n_components=None):
    """
    Transform the data using pca
    ---
    @param data: a dataframe
    """
    return PCA(n_components=n_components).fit_transform(data)


def normalize(data):
    """
    Normalize the Data between 0 and 1
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with normalized values
    """
    return (data - data.min()) / (data.max() - data.min())


def standardize(data):
    """
    Standardize the data against its standard deviation
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with standardized values
    """
    return (data - data.min()) / data.std()


def log_transform(data):
    """
    normalize and log transform teh dataframe
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with log transformed values
    """
    return log(normalize(data))
