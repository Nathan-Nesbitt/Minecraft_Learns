"""
    Defines KNN Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.cluster import KMeans as KMeansModel


class KMeans(ClassificationModel):
    """
    Model class for KMeans Classification
    """

    def __init__(self, n_clusters=3, pca=False):
        """
        @param n_clusters: the number of clusters to create
        @param pca: boolean defining if pca will be run
        """
        super().__init__(pca)
        self.n_clusters = n_clusters
        self.internal_model = KMeansModel(n_clusters=n_clusters)

    def set_parameters(self, params):
        """
        Set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        super().set_parameters(params)
        # set k if necessary and add number of clusters to model
        if "k" in params.keys():
            self.n_clusters = params["k"]
            self.internal_model.set_params(**{"n_clusters": params["k"]})

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
        return super().predict(X)
