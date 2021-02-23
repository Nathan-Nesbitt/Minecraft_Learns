"""
    Add some common methods for Machine Learning
"""

from numpy import power, sqrt, log
from numpy import sum as np_sum

from pandas import concat, DataFrame

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


ALMOST_ZERO = 0.000001


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
    pca = PCA(n_components=n_components)
    return pca.fit_transform(mean_normalization(data))


def mean_normalization(data):
    """
    Normalize the Data between -1 and 1 with mean 0
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with normalized values
    """
    if (data.std() != 0).all():
        return (data - data.mean()) / data.std()
    else:
        return (data - data.mean()) / (data.std() + ALMOST_ZERO)


def normalize(data):
    """
    Normalize the Data between 0 and 1
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with normalized values
    """
    if ((data.max() - data.min()) != 0).all():
        return (data - data.min()) / (data.max() - data.min())
    else:
        return (data - data.min()) / (data.max() - data.min() + ALMOST_ZERO)


def standardize(data):
    """
    Standardize the data against its standard deviation
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with standardized values
    """
    if (data.std() != 0).all():
        return (data - data.min()) / data.std()
    else:
        return (data - data.min()) / (data.std() + ALMOST_ZERO)


def label_encoding(data):
    """
    encode the data at the following columns and save the label encoder
    ---
    @param data: a dataframe
    """
    # get the columns to encode
    encode_cols = []
    if data.ndim == 1:
        encode_cols = data.name
    else:
        encode_cols = data.select_dtypes(include=["object"]).columns

    # encode the labels
    label_encoder = LabelEncoder().fit(data[encode_cols])
    temp = label_encoder.transform(data[encode_cols])
    data.loc[:, encode_cols] = DataFrame(temp, columns=encode_cols)

    # return the new data
    return label_encoder, data


def encode_labels(label_encoder, data):
    """
    encode the data at the following columns and save the label encoder
    ---
    @param data: a dataframe
    """
    return label_encoder.transform(data)

def log_transform(data):
    """
    normalize and log transform teh dataframe
    ---
    @param data: a dataframe
    ---
    outputs a new dataframe with log transformed values
    """
    return log(normalize(data))


def is_dataframe(data):
    return isinstance(data, DataFrame)


def get_ith_column(data, i=0):
    """
    get the ith column of a dataframe or array
    ---
    @param data: a dataframe
    @param i: the column number
    """
    if data.ndim == 1:
        return data

    if is_dataframe(data):
        return data[data.columns[0]], data.columns[0]
    else:
        return data[:, 0], "0"
