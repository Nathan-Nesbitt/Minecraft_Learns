"""
    Defines a model to be trained using the data and a model name.
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from sklearn.decomposition import PCA
from numpy import log
from sklearn.model_selection import cross_val_score


class Model:
    """
    Generic Abstract Model Class all models inherit fromx
    ---
    TODO: convert to Abstract Base Class with abc
    """
    def __init__(self, pca=False):
        self.internal_model = None
        self.pca = pca
        self.score = [0,0,0,0,0]

    def process_data(self, data):
        """
        Process the data
        """
        if self.pca:
            return self._pca(self._standardize(data))
        else:
            return self._standardize(data)

    def train(self):
        """
        Train the Model
        """
        pass

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        pass
    
    def evaluate(self, y):
        return self.score.mean()

    def _pca(self, data, n_components=None):
        """
        Transform the data using pca
        ---
        @param data: a dataframe
        """
        print(data)
        return PCA(n_components=n_components).fit_transform(data)

    def _normalize(self, data):
        """
        Normalize the Data between 0 and 1
        ---
        @param data: a dataframe
        ---
        outputs a new dataframe with normalized values
        """
        return (data - data.min())/(data.max() - data.min())

    def _standardize(self, data):
        """
        Standardize the data against its standard deviation
        ---
        @param data: a dataframe
        ---
        outputs a new dataframe with standardized values
        """
        return (data - data.min())/data.std()

    def _log_transform(self, data):
        """
        normalize and log transform teh dataframe
        ---
        @param data: a dataframe
        ---
        outputs a new dataframe with log transformed values
        """
        return log(self._normalize(data))

    def _evaluate(self, X, y):
        """
        Score the model using the default values
        """
        self.score = cross_val_score(self.internal_model, X, y, cv=5)

    def set_X(self, X):
        """
        Set X to an input value
        ---
        @param X: a dataframe with n predictor observations
        """
        self.X = X
    
    def set_y(self, y):
        """
        Set y to an input value
        ---
        @param y: a series with n response observations
        """
        self.y = y
