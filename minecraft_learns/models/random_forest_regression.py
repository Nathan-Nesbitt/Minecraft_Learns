"""
    Defines Random Forest Model used for Regression 
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


from .generic.regression_model import RegressionModel
from sklearn.ensemble import RandomForestRegressor as RandomForest


class RandomForestRegressor(RegressionModel):
    """
    Model class for Random Forest Regression
    """
    def __init__(self, pca=False):
        super.__init__(pca)
        self.internal_model = RandomForest(
            bootstrap=True, oob_score=True, max_samples=0.3
        )

    def process_data(self, X, y):
        """
        Trees do not need data processing, set X and y
        ---
        @param X: a dataframe with n predictor observations
        @param y: a series with n response observations
        """
        super().set_X(X)
        super().set_y(y)

    def train(self):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        self.internal_model = self.internal_model.fit(self.X, self.y)
        super()._evaluate(self.X, self.y)

    def predict(self, X):
        """
        Predict the response variable y for the input data X
        ---
        @param X: a 2D data matrix of n observations and m predictors
        """
        predicted_y = self.internal_model.predict(X)
        super()._evaluate(X, predicted_y)
        return predicted_y

    def feature_importance(self):
        return self.internal_model.feature_importances_
