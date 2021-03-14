"""
    Defines Support Vector Machines used for Regression and Classification

    Written By: Kathryn Lecha
    Date: 2021-03-13

    References:
    https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
"""


from .common import one_hot_encode
from .generic.regression_model import RegressionModel
from .generic.classification_model import ClassificationModel

from numpy import argmax, zeros, max as npmax
from sklearn.svm import SVC, SVR


def get_best_c(svm, X_train, y_train, X_test, y_test, c_options, c_only=False):
    """
    determine which c is the best
    """
    test_svm = svm
    accuracy_array = zeros(len(c_options))
    for i in range(0,len(c_options)):
        # set the new c
        c = c_options[i]
        test_svm.set_params(**{"C": c})

        test_svm.fit(X_train, y_train)

        # get the accuracy
        accuracy_array[i] = test_svm.score(X_test, y_test)
    
    best_c = c_options[argmax(accuracy_array)]
    best_accuracy = npmax(accuracy_array)
    return best_c, best_accuracy, accuracy_array


class SVMClassification(ClassificationModel):
    """
    class for SVM classficication
    ---
    Wrapper class for 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(
            self, kernel="rbf", degree=3, pca=False, one_hot_encode=False
        ):
        """
        @param kernel: kernel for 
        @param one_hot_encode: False OR a list of column names to encode
        """
        super().__init__(pca)
        self.kernel = kernel
        self.internal_model = SVC(kernel=kernel, degree=degree, max_iter=4000)

    def set_parameters(self, params):
        """
        Set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        super().set_parameters(params)

        # reset one hot encoding
        if "one_hot_encode" in params.keys():
            self.one_hot_encode = params["one_hot_encode"]

        # set model parameters
        if "kernel" in params.keys():
            self.kernel = params["kernel"]
            self.internal_model.set_params(**{"kernel": params["kernel"]})
        if "C" in params.keys():
            self.internal_model.set_params(**{"C": params["C"]})
    
    def process_data(self, X, y):
        """
        set X and y
        """
        if self.one_hot_encode:
            super().set_X(one_hot_encode(X, self.one_hot_encode))
        else:
            super().set_X(super().process_data(X))

        super().set_y(super().process_data(y))
    
    def get_support_vectors(self):
        """return the support vectors of the model"""
        return self.internal_model.support_vectors_

    def get_n_support_vectors(self):
        return self.internal_model.n_support_
    
    def set_c(self, C):
        self.internal_model.set_params(**{"C": C})

    def set_best_c(self):
        c = [0.1, 1, 5, 10, 20, 50, 100]
        c = get_best_c(
            self.internal_model, self.X, self.y, self.X, self.y, c, c_only=True
        )
        self.set_c(c)


class SVMRegression(RegressionModel):
    """
    class for SVM Regression
    ----
    Wrapper class for:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    def __init__(
            self, kernel="rbf", degree=3, pca=False, one_hot_encode=False
        ):
        """
        @param kernel: kernel for 
        @param one_hot_encode: False OR a list of column names to encode
        """
        super().__init__(pca)
        self.kernel = kernel
        self.internal_model = SVR(kernel=kernel, degree=degree, max_iter=4000)

    def set_parameters(self, params):
        """
        Set the parameters of the model
        ---
        @param params: dictionary of parameters to set
        """
        if params is None:
            return
        super().set_parameters(params)

        # reset one hot encoding
        if "one_hot_encode" in params.keys():
            self.one_hot_encode = params["one_hot_encode"]

        # set model parameters
        if "kernel" in params.keys():
            self.kernel = params["kernel"]
            self.internal_model.set_params(**{"kernel": params["kernel"]})
        if "C" in params.keys():
            self.internal_model.set_params(**{"C": params["C"]})
    
    def process_data(self, X, y):
        """
        set X and y
        """
        if self.one_hot_encode:
            super().set_X(one_hot_encode(X, self.one_hot_encode))
        else:
            super().set_X(super().process_data(X))

        super().set_y(super().process_data(y))
    
    def get_support_vectors(self):
        """return the support vectors of the model"""
        return self.internal_model.support_vectors_

    def get_n_support_vectors(self):
        return self.internal_model.n_support_
    
    def set_c(self, C):
        self.internal_model.set_params(**{"C": C})

    def set_best_c(self):
        c = [0.1, 1, 5, 10, 20, 50, 100]
        c = get_best_c(
            self.internal_model, self.X, self.y, self.X, self.y, c, c_only=True
        )
        self.set_c(c)
