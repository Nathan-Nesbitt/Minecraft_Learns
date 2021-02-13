"""
    Defines KNN Model used for Classification 

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from generic.classification_model import ClassificationModel
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
        KMeans needs 
        """
        self.X = super().process_data(X)
        self.y = super().process_data(y)

    def train(self):
        """
        Train the Model
        """
        pass

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        pass
    
    def evaluate(self):
        pass

    def get_centroids(self):
        """
        Return the centroids of the model
        """
        return self.internal_model.cluster_centers_
