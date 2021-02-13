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
    def __init__(self, k, pca=True):
        super.__init__(pca)
        self.k = k
        self.internal_model = KMeansModel(n_clusters=k)

    def process_data(self, X, y):
        """
        Standardize the data and do PCA if needed
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        self.X = super().process_data(X)
        self.y = super().process_data(y)

    def train(self):
        """
        Train the Model
        ---
        sets the internal model to the trained model
        sets the evalutation to the fit
        """
        self.internal_model = self.internal_model.fit(self.X, self.y)
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
        return preformance of model
        """
        return self.score
    
    def _evaluate(self, X, y):
        """
        Score the model using the default values
        """
        self.score = self.internal_model.score(X, y)
