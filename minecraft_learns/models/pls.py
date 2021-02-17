"""
    Defines PLS Regression - this is more refined than PCA Linear Regression
    This can be used to predict multiple variables by condensing into one
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


from .generic.regression_model import RegressionModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder

from pandas import concat, DataFrame


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

    def process_data(self, X, y):
        """
        PLS will handle PCA and scalling automatically
        ---
        @param X: a dataframe with n observations of m predictors
        @param y: a dataframe with n observations of t response targets
        """
        if self.one_hot_encode:
            self.set_X(self._one_hot_encode(X))
        else:
            self.set_X(X)
        self.set_y(y)

    def _process_data(self, data):
        """
        run interactions and encoding as necessary
        ---
        @param data: dataframe of n observervations
        """
        if self.one_hot_encode:
            data = self._one_hot_encode(data)
        return data

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
        predicted_y = self.internal_model.predict(self._process_data(X))
        super()._evaluate(X, predicted_y)
        return predicted_y
    
    def get_coefficents(self):
        return self.internal_model.coef_

    def equation(self):
        """
        returns a string of the equation used
        """
        equation_string = ""

        # add the coefficents to the string 
        coefficents = self.internal_model.coef_
        for i in range(0,len(self.X.columns)):
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

    def _one_hot_encode(self, X):
        """
        one hot encode the data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        # encode the columns
        encoded = X[self.one_hot_encode]
        encoded = OneHotEncoder(sparse=False).fit_transform(encoded)
        encoded = DataFrame(encoded, index=X.index)
        # remove the duplicates and concatenate
        return concat([X.drop(self.one_hot_encode, axis=1), encoded], axis=1)