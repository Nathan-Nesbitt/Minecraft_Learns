"""
    Defines PLS Regression - this is more refined than PCA Linear Regression
    This can be used to predict multiple variables by condensing into one

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from ..common import one_hot_encode
from .generic.regression_model import RegressionModel

from sklearn.cross_decomposition import PLSRegression


class PLSRegressor(RegressionModel):
    """
    Model class for PLS Regression
    """

    def __init__(self, n_components=3, one_hot_encode=False):
        """
        Initalize PLS Regression
        ---
        @param n_components: number of components to keep in PCA
        @param one_hot_encode: False OR a list of column names to encode
        """
        super().__init__(pca=False)
        self.one_hot_encode = one_hot_encode
        self.internal_model = PLSRegression(n_components=n_components)

    def set_parameters(self, params):
        """
        Set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        super().set_parameters(params)

        if "one_hot_encode" in params.keys():
            self.one_hot_encode = params["one_hot_encode"]

    def process_data(self, X, y):
        """
        PLS will handle PCA and scalling automatically
        ---
        @param X: a dataframe with n observations of m predictors
        @param y: a dataframe with n observations of t response targets
        """
        self.set_X(self._process_data(X))
        self.set_y(y)

    def _process_data(self, data):
        """
        run interactions and encoding as necessary
        ---
        @param data: dataframe of n observervations
        """
        if self.one_hot_encode:
            data = one_hot_encode(data, self.one_hot_encode)
        return data

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = self._process_data(X)
        return super().predict(X)

    def get_coefficents(self):
        return self.internal_model.coef_

    def equation(self):
        """
        returns a string of the equation used
        """
        equation_string = ""

        # add the coefficents to the string
        coefficents = self.internal_model.coef_
        for i in range(0, len(self.X.columns)):
            equation_string += coefficents[i] + "*" + self.X.columns[i]
            if i < (len(self.X.columns) - 1):
                equation_string += " + "

        return equation_string

    def transform_data(self, X):
        """
        Transform the data X into the new dimension
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        return self.internal_model.transform(X)

    def inverse_transform_data(self, X):
        """
        Revert X from new dimension to original
        ---
        @param X: a 2D data matrix of transformed data
        """
        return self.internal_model.inverse_transform(X)
