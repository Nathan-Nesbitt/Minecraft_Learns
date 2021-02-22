"""
    Defines DecisionTree Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from .generic.regression_model import RegressionModel

from sklearn.tree import DecisionTreeClassifier as DecisionTreeClas
from sklearn.tree import DecisionTreeRegressor as DecisionTreeReg
from ..graphing import plot_decision_tree


class DecisionTreeClassifier(ClassificationModel):
    """
    Model class for Decision Tree Classification
    """

    def __init__(self, pca=False):
        super().__init__(pca)
        self.internal_model = DecisionTreeClas()

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

    def plot(self, tree=True, location=None):
        if tree:
            plot_decision_tree(
                self.internal_model, columns=self.X.columns, location=location
            )
        else:
            super().plot(location)


class DecisionTreeRegression(RegressionModel):
    """
    Model class for Decision Tree Regression
    """

    def __init__(self, pca=False):
        super().__init__(pca)
        self.internal_model = DecisionTreeReg()

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

    def plot(self, tree=True, location=None):
        if tree:
            plot_decision_tree(
                self.internal_model, columns=self.X.columns, location=location
            )
        else:
            super().plot(location)
