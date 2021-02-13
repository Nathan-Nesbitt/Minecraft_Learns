"""
    Defines DecisionTree Model used for Classification 
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(ClassificationModel):
    """
    Model class for Decision Tree Classification
    """
    def __init__(self, pca=False):
        super.__init__(pca)
        self.internal_model = DecisionTreeClassifier()

    def process_data(self, X, y):
        """
        Decision Trees do not need data processing, set X and y
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        self.X = X
        self.y = y

    def train(self):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
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
