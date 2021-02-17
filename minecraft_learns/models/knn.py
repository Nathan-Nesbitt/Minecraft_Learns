"""
    Defines KNN Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.neighbors import KNeighborsClassifier


class KNN(ClassificationModel):
    """
    Model class for KNN Classification
    """

    def __init__(self, k):
        super().__init__()
        self.internal_model = KNeighborsClassifier(
            n_neighbors=k, weights="distance"
            )

    def process_data(self, X, y):
        """
        Standardize the data and do PCA if needed
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        super().set_X(super().process_data(X))
        super().set_y(super().process_data(y))

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = super().process_data(X)
        predicted_y = super().predict(X)
        return predicted_y
