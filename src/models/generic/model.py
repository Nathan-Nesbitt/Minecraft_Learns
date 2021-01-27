"""
    Defines a model to be trained using the data and a model name.
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


class Model:
    """
    Generic Abstract Model Class all models inherit from
    ---
    TODO: convert to Abstract Base Class with abc
    """
    def __init__(self, goal, pca=False):
        self.goal = goal
        self.internal_model = None
        self.pca = True

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

    def _pca(self, data):
        pass

    def _normalize(self, data):
        pass

    def _split_xy(self, data):
        pass
