import pytest
from numpy import array, log
from pandas import DataFrame

from minecraft_learns.models.random_forest import (RandomForestClassifier,
                                                   RandomForestRegressor)

from sklearn.ensemble import RandomForestRegressor as RandomForestReg
from sklearn.ensemble import RandomForestClassifier as RandomForestClas


def test_classifier():
    classifier = RandomForestClassifier()
    assert type(classifier.internal_model) == RandomForestClas

    X = array([[0,1,2,3],[2,0,2,-1],[1,3,4,1]])
    y = array([0,1,1])

    classifier.process_data(X, y)
    classifier.train()
    assert classifier.evaluate() is not None

    assert classifier.predict().shape == y.shape
    assert classifier.predict_probablity(X).shape[0] == y.shape[0]
    assert classifier.predict_probablity(X).shape[1] == 2

    assert classifier.feature_importance() is not None

    assert 0 <= classifier.misclassification_rate(X, y) <= 1
    assert 0 <= classifier.accuracy(X, y) <= 1
    assert 0 <= classifier.precision(X, y) <= 1

def test_regressor():
    regressor = RandomForestRegressor()
    assert type(regressor.internal_model) == RandomForestReg

    X = array([[0,1,2,3],[2,0,2,-1],[1,3,4,1]])
    y = array([0,2,1])

    regressor.process_data(X, y)
    regressor.train()
    assert regressor.evaluate() is not None
    assert regressor.predict().shape == y.shape

    assert -1 <= regressor.r_square() <= 1
    assert 0 <= regressor.mse()

    assert regressor.feature_importance() is not None


