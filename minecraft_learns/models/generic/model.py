"""
    Defines a model to be trained using the data and a model name.

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from ...common import pca, standardize
from ...errors import UnProcessedData

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
        self.score = [0, 0, 0, 0, 0]

    def set_parameters(self, params):
        """
        set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params.haskey("pca"):
            self.pca = params["pca"]

    def process_data(self, data):
        """
        Process the data
        """
        if self.pca:
            return pca(standardize(data))
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
            raise UnProcessedData("train")

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        predicted_y = self.internal_model.predict(X)
        self._evaluate(X, predicted_y)
        return predicted_y

    def evaluate(self, y):
        return self.score.mean()

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
