"""
Defines a classification model to be trained

Written By: Kathryn Lecha
Edit-Date: 2021-02-13
"""


from .model import Model
from ...graphing import scatter_groups
from ...common import is_dataframe

from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from matplotlib import pyplot


class ClassificationModel(Model):
    """
    Generic Abstract Classification Model Class all classification models to
    inherit from
    """

    def __init__(self, pca=False):
        super().__init__(pca)

    def predict_probablity(self, X):
        """
        Predict the probablity of label y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        return self.internal_model.predict_proba(X)

    def misclassification_rate(self, test_X, test_y):
        """
        get the misclassification rate of the model
        ---
        @param test_X: dataframe of n test predictor observations
        @param test_y: dataframe of n test response observations
        """
        return 1 - self.accuracy(test_X, test_y)

    def accuracy(self, test_X, test_y):
        """
        Return the accuracy of the model
        ---
        @param test_X: dataframe of n test predictor observations
        @param test_y: dataframe of n test response observations
        """
        predicted_y = self.internal_model.predict(test_X)
        return accuracy_score(test_y, predicted_y, normalize=True)

    def precision(self, test_X, test_y):
        """
        Return the accuracy of the model
        ---
        @param test_X: dataframe of n test predictor observations
        @param test_y: dataframe of n test response observations
        """
        predicted_y = self.internal_model.predict(test_X)
        return average_precision_score(test_y, predicted_y)

    def plot_precision_recall(self, test_X, test_y):
        """
        plot the precesion recall curve
        ---
        @param test_X: dataframe of n test predictor observations
        @param test_y: dataframe of n test response observations
        """
        average_precision = self.precision(test_X, test_y)
        plot = plot_precision_recall_curve(self.internal_model, test_X, test_y)
        title = "2-class Precision-Recall curve: AP={0:0.2f}"
        plot.ax_.set_title(title.format(average_precision))

    def plot(self, location=None):
        """
        plot the groups
        """
        if is_dataframe(self.X):
            scatter_groups(
                self.X[self.X.columns[0]], self.X[self.X.columns[1]],
                self.predict(self.X),
                "Groups Found", self.X.columns[0], self.X.columns[1],
                location
            )
        else:
            scatter_groups(
                self.X[:,0], self.X[:,1], self.predict(self.X),
                "Groups Found", "0", "1",
                location
            )
