"""
Defines a regression model to be trained

Written By: Kathryn Lecha
Date: 2021-01-26
"""


from .model import Model
from ...common import is_dataframe
from ...graphing import scatter

from numpy import square, subtract


class RegressionModel(Model):
    """
    Generic Abstract Regression Model Class all classification models to
    inherit from
    """

    def __init__(self, pca=False):
        super().__init__(pca)

    def r_square(self):
        """Return r square score"""
        return self.internal_model.score()

    def mse(self):
        """
        evaluate the preformance of the model using MSE
        """
        return square(subtract(self.y.values, self.predict(self.X))).mean()

    def plot(self, location=None):
        if is_dataframe(self.X):
            scatter(
                self.X[self.X.columns[0]], self.predict(self.X),
                "Predicted Regression", self.X.columns[0], "Predicted Y",
                location
            )
        else:
            scatter(
                self.X[:,0], self.predict(self.X),
                "Predicted Regression", "x", "Predicted Y",
                location
            )
