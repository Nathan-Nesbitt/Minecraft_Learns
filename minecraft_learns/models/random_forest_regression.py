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
        super().__init__(pca)
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

    def feature_importance(self):
        return self.internal_model.feature_importances_
