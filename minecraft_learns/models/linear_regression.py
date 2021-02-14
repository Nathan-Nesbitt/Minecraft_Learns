"""
    Defines Linear Regression
    Options for Linear Regression:
        - linear regression: pca = False
        - PCA LR: pca = True (default)
        - one hot encoding: one_hot_encode=[list of column names]
        - interations: interactions=[list of column names]
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


from .generic.regression_model import RegressionModel
from sklearn.linear_model import LinearRegression as LinearRegressionModel
from sklearn.preprocessing import OneHotEncoder

from pandas import concat
from numpy import square, subtract


class LinearRegression(RegressionModel):
    """
    Model class for Linear Regression
    """
    def __init__(self, pca=True, one_hot_encode=False, interactions=False):
        """
        @param one_hot_encode: False OR a list of column names to encode
        @param interactions: False OR a list of column names to interaction
        """
        super.__init__(pca)
        self.interactions = interactions
        self.one_hot_encode = one_hot_encode
        self.internal_model = LinearRegressionModel()

    def process_data(self, X, y):
        if self.one_hot_encode:
            X = self._one_hot_encode(X)
        super().set_X(super().process_data(X))
        super().set_X(super().process_data(y))

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
    
    def mse(self):
        """
        evaluate the preformance of the model using MSE
        """
        return square(subtract(self.y.values, self.predict(self.X))).mean()

    def _interact(self, X):
        """
        add interactions to X for the columns in self.columns
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        # for each pair of interaction columns, create a new product column
        for column_1 in self.interactions:
            for column_2 in self.interactions:
                if column_1 != column_2:
                    column_name = "" + column_1 + "*" + column_2
                    X[column_name] = X[column_1] * X[column_2]
        return X

    def _one_hot_encode(self, X):
        """
        one hot encode the data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        # encode the columns
        encoded = X[self.one_hot_encode]
        encoded = OneHotEncoder(sparse=False).fit_transform(encoded)
        # remove the duplicates and concatenate
        return concat([X.drop(self.one_hot_encode), encoded], axis=1)
    
    def get_intercept(self):
        return self.internal_model.intercept_
    
    def get_coefficents(self):
        return self.internal_model.coef_

    def equation(self):
        """
        returns a string of the equation used
        """
        equation_string = "" + self.internal_model.intercept_

        # add the coefficents to the string 
        coefficents = self.internal_model.coef_
        for i in range(0,len(self.X.columns)):
            equation_string += " + " + coefficents[i] + "*" + self.X.columns[i]

        return equation_string
