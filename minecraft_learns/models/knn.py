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

    def train(self):
        """
        Train the Model and set the model to trained model
        evaluate based on training
        """
        self.internal_model = self.internal_model.fit(self.X, self.y)
        self._evaluate(self.X, self.y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        X = super().process_data(X)
        predicted_y = self.internal_model.predict(X)
        self._evaluate(X, predicted_y)
        return predicted_y

    def predict_probablity(self, X):
        """
        Predict the probablity of label y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        return self.internal_model.predict_proba(X)
