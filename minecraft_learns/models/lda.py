"""
    Defines LDA Model used for Classification

    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from .generic.classification_model import ClassificationModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(ClassificationModel):
    """
    Model class for LDA Classification
    """

    def __init__(self, pca=True):
        super().__init__(pca)
        self.internal_model = LinearDiscriminantAnalysis()

    def process_data(self, X, y):
        """
        set X and y to standardized data
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

    def mean(self):
        """
        return the overall mean of the model
        """
        return self.internal_model.xbar_

    def class_means(self):
        """
        return the means of each class
        """
        return self.internal_model.means_

    def get_intercept(self):
        return self.internal_model.intercept_

    def get_coefficents(self):
        return self.internal_model.coef_

    def equation(self):
        """
        returns a string of the equation used
        """
        equation_string = "" + self.internal_model.intercept_

        # add the coefficents to the string
        coefficents = self.internal_model.coef_
        for i in range(0, len(self.X.columns)):
            equation_string += " + " + coefficents[i] + "*" + self.X.columns[i]

        return equation_string
