import pytest
from numpy import array, log
from pandas import DataFrame

from minecraft_learns.common import (euclidean_distance, mean_zero_normalize,
                                    is_dataframe, standardize, log_transform)


def test_euclidean_distance():
    a = array([[0,1,2,3],[2,0,2,-1]])
    b = array([1,3,4,1])

    dist = euclidean_distance(a, b)

    assert dist.shape == (2,)


def test_mean_zero_normalize():
    a = array([[0,1,2,3],[2,0,2,-1]])

    b = mean_zero_normalize(a)

    assert a.shape == b.shape
    assert b.mean() == 0


def test_standardize():
    a = array([[0,1,2,3],[2,0,2,-1]])

    b = standardize(a)

    assert a.shape == b.shape


def test_is_dataframe():
    a = array([[0,1,2,3],[2,0,2,-1]])
    b = DataFrame([[0,1,2,3],[2,0,2,-1]])

    assert not is_dataframe(a)
    assert is_dataframe(b)


def test_log_transform():
    a = array([[0,1,2,3],[2,0,2,-1]])
    b = log_transform(a)

    assert a.shape == b.shape
    assert a.min() >= b.min()
