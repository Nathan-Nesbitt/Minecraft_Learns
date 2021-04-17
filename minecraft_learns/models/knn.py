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

    def __init__(self, n_neighbors=3, pca=False):
        """
        Initalize the model
        ----
        @param n_neighbors: the number of neighbors to consider
        """
        super().__init__(pca)
        self.n_neighbors = n_neighbors
        self.internal_model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance"
        )

    def set_parameters(self, params):
        """
        Set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        super().set_parameters(params)

        # set k if necessary and add number of neighbors to model
        if "k" in params.keys():
            self.n_neighbors = params["k"]
            self.internal_model.set_params(**{"n_neighbors": params["k"]})

    def process_data(self, X, y):
        """
        Standardize the data and do PCA if needed
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
        predicted_y = super().predict(X)
        return predicted_y
