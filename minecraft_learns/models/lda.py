"""
    Defines LDA Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(ClassificationModel):
    """
    Model class for LDA Classification
    """

    def __init__(self, pca=True):
        super().__init__(pca)
        self.internal_model = LinearDiscriminantAnalysis()

    def process_data(self, X, y):
        """
        set X and y to standardized data
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        super().set_X(super().process_data(X))
        super().set_y(y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = super().process_data(X)
        return super().predict(X)

    def mean(self):
        """
        return the overall mean of the model
        """
        return self.internal_model.xbar_

    def class_means(self):
        """
        return the means of each class
        """
        return self.internal_model.means_

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
        for i in range(0, len(self.X.columns)):
            equation_string += " + " + coefficents[i] + "*" + self.X.columns[i]

        return equation_string
