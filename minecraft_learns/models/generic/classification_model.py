"""
Defines a classification model to be trained
    
Written By: Kathryn Lecha
Date: 2021-01-26
"""


from model import Model


class ClassificationModel(Model):
    """
    Generic Abstract Classification Model Class all classification models to
    inherit from
    """
    def __init__(self, pca=False):
        super().__init__(pca)

