"""
Defines a regression model to be trained

Written By: Kathryn Lecha
Date: 2021-01-26
"""


from .model import Model
from ...common import get_ith_column, is_dataframe
from ...graphing import scatter


class RegressionModel(Model):
    """
    Generic Abstract Regression Model Class all classification models to
    inherit from
    """

    def __init__(self, pca=False):
        super().__init__(pca)

    def r_square(self):
        """Return r square score"""
        return self.internal_model.score(self.X, self.y)

    def mse(self):
        """
        evaluate the preformance of the model using MSE
        """
        if is_dataframe(self.y):
            return ((self.y.values - self.predict()) ** 2).mean()
        else:
            return ((self.y - self.predict()) ** 2).mean()

    def plot(self, location=None):
        x, x_label = get_ith_column(self.X, 0)
        y = get_ith_column(self.predict(self.X), 0)[0]
        scatter(x, y, "Predicted Regression", x_label, "y", location)
