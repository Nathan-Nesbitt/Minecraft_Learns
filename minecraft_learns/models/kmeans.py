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

    def __init__(self, k, pca=False):
        super().__init__(pca)
        self.k = k
        self.internal_model = KMeansModel(n_clusters=k)

    def process_data(self, X, y):
        """
        Standardize the data and do PCA if needed
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        super().set_X(super().process_data(X))
        super().set_y(super().process_data(y))

    def train(self):
        """
        Train the Model
        ---
        sets the internal model to the trained model
        sets the evalutation to the fit
        """
        self.internal_model = self.internal_model.fit(self.X, self.y)
        super()._evaluate(self.X, self.y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = super().process_data(X)
        predicted_y = self.internal_model.predict(X)
        super()._evaluate(X, predicted_y)
        return predicted_y

    def predict_probablity(self, X):
        """
        Predict the probablity of label y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        return self.internal_model.predict_proba(X)
