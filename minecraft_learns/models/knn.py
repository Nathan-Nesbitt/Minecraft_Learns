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
    def __init__(self, k, pca):
        super.__init__(pca)
        self.internal_model = KNeighborsClassifier(
            n_neighbors=k, weights="distance"
            )

    def process_data(self, X, y):
        """
        Process the data
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        self.X = super().process_data(X)
        self.y = super().process_data(y)

    def train(self):
        """
        Train the Model
        """
        self.internal_model.fit(self.X, self.y)
        self._evaluate(self.X, self.y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = super()._standardize(X)
        predicted_y = self.internal_model.predict(X)
        self._evaluate(X, predicted_y)
        return predicted_y
    
    def evaluate(self):
        """
        evaluate the preformance of the model using MSE
        """
        return self.score
    
    def _evaluate(self, X, y):
        """
        Score the model using the default values
        """
        self.score = self.internal_model.score(X, y)
