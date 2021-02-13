"""
    Defines Random Forest Model used for Regression 
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


from .generic.regression_model import RegressionModel
from sklearn.linear_model import LinearRegression as LinearRegressionModel
from sklearn.preprocessing import OneHotEncoder


class LinearRegression(RegressionModel):
    """
    Model class for Linear Regression
    """
    def __init__(self, pca=True, one_hot_encode=False):
        """
        @param interactions: False OR a list of column names to interact
        """
        super.__init__(pca)
        self.one_hot_encode = one_hot_encode
        self.internal_model = LinearRegressionModel()

    def process_data(self, X, y):
        if self.one_hot_encode:
            X = self._one_hot_encode(X)
        self.X = super().process_data(X)
        self.y = super().process_data(y)

    def train(self):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        self.internal_model = self.internal_model.fit(self.X, self.y)
        self._evaluate(self.X, self.y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        if self.one_hot_encode:
            X = self._one_hot_encode(X)
        X = super().process_data(X)

        predicted_y = self.internal_model.predict(X)
        super()._evaluate(X, predicted_y)
        return predicted_y
    
    def evaluate(self):
        """
        evaluate the preformance of the model using MSE
        """
        return self.score

    def _one_hot_encode(self, X):
        """
        one hot encode the data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        enc = OneHotEncoder(sparse=False)
        return enc.fit_transform(X)
