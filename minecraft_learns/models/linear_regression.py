"""
    Defines Random Forest Model used for Regression 
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""


from generic.regression_model import RegressionModel


class LinearRegression(RegressionModel):
    """
    Model class for Linear Regression
    """
    def __init__(self, pca=True, interactions=False):
        super.__init__(pca)
        self.interactions = interactions

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
