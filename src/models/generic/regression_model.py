"""
Defines a classification model to be trained
    
Written By: Kathryn Lecha
Date: 2021-01-26
"""


from model import Model


class RegressionModel(Model):
    """
    Generic Abstract Regression Model Class all classification models to
    inherit from
    """
    def __init__(self, goal, pca=False):
        super.__init__(self, goal, pca)

