"""
    Defines a model to be trained using the data and a model name.
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""

from sklearn.decomposition import PCA


class Model:
    """
    Generic Abstract Model Class all models inherit from
    ---
    TODO: convert to Abstract Base Class with abc
    """
    def __init__(self, goal, pca=False):
        self.goal = goal
        self.internal_model = None
        self.pca = True

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

    def _pca(self, data, n_components=None):
        """
        Transform the data using pca
        ---
        @param data: a dataframe
        ---
        Outputs:
            pca: PCA object for transformations
            pca_data: transformed data
        """
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data)
        return pca, pca_data

    def _normalize(self, data):
        """
        Normalize the Data between 0 and 1
        ---
        @param data: a dataframe
        ---
        outputs a new dataframe with normalized values
        """
        return (data - data.min())/(data.max() - data.min())

    def _standardize(self, data):
        """
        Standardize the data against its standard deviation
        ---
        @param data: a dataframe
        ---
        outputs a new dataframe with standardized values
        """
        return (data - data.min())/data.std()

    def _split_xy(self, data):
        pass
