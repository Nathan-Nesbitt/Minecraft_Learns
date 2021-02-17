"""
    Defines DecisionTree Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


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
        super().set_X(X)
        super().set_y(y)

    def feature_importance(self):
        return self.internal_model.feature_importances_

    def plot_decision_tree(self):
        plot_tree(self.internal_model, feature_names=self.X.columns)
