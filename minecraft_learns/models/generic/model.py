"""
    Defines a model to be trained using the data and a model name.

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from ...common import pca, standardize
from ...errors import UnProcessedData, ModelNotFit

from joblib import dump, load

from sklearn.model_selection import cross_val_score
from sklearn.exceptions import NotFittedError
from numpy import nan


class Model:
    """
    Generic Abstract Model Class all models inherit fromx
    """

    def __init__(self, pca=False):
        self.internal_model = None
        self.pca = pca
        self.score = nan

    def set_parameters(self, params):
        """
        set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        if "pca" in params.keys():
            self.pca = params["pca"]

    def process_data(self, data):
        """
        Process the data
        """
        if self.pca:
            return pca(data)
        else:
            return standardize(data)

    def train(self):
        """
        Train the Model
        sets the internal model to the trained model
        sets the evalutation to the crossvalidation score of the fit
        """
        try:
            self.internal_model = self.internal_model.fit(self.X, self.y)
            self._evaluate(self.X, self.y)
        except(AttributeError):
            raise UnProcessedData()

    def predict(self, X=None):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        if X is None:
            X = self.X

        try:
            # if X is one dimensional, reshape it
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # predict and evaluate
            predicted_y = self.internal_model.predict(X)
            self._evaluate(X, predicted_y)
            return predicted_y
        except NotFittedError:
            raise ModelNotFit()

    def evaluate(self):
        return self.score

    def _evaluate(self, X, y):
        """
        Score the model using the default values
        """
        if len(X) > 5:
            self.score = cross_val_score(self.internal_model, X, y).mean()

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
    
    def save_model(self, filename):
        dump(self.internal_model, filename)
    
    def load_model(self, filename):
        self.internal_model = load(filename)
