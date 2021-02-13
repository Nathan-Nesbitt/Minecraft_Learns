"""
    Defines LDA Model used for Classification 
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from generic.classification_model import ClassificationModel


class LDA(ClassificationModel):
    """
    Model class for LDA Classification
    """
    def __init__(self, pca):
        super().__init__(pca)

    def process_data(self, data):
        pass

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
