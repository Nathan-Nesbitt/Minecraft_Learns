import pytest
from numpy import array, log
from pandas import DataFrame

from minecraft_learns.common import *


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


def test_interact():
    df = DataFrame([[0,1,2,3],[2,0,2,-1]], columns=["a","b","c","d"])
    df = interact(df, ["a", "b"])
    assert (df.columns == ["a","b","c","d","a*b"]).all()


def test_one_hot_encode():
    df = DataFrame([[0,1,2],[2,0,2]], columns=["a","b","c"])
    df = one_hot_encode(df, ["a"])
    print(df.columns)
    assert (df.columns == ["b","c",0,1]).all()

def test_pca():
    a = DataFrame([[0,1,2,3],[2,0,2,-1],[0,1,2,3],[2,0,2,-1]])
    
    assert pca(a).shape[1] == a.shape[1]
    assert pca(a,2).shape[1] == 2

def test_normalize():
    a = array([[0,1,2,3],[2,0,2,-1]])

    b = normalize(a)

    assert a.shape == b.shape
    assert b.min() == 0
    assert b.max() == 1

def test_label_encoding():
    columns=["a","b","label"]
    df = DataFrame([[0,1,"a"],[2,0,"b"]], columns=columns)
    beginshape = df.shape

    le, df = label_encoding(df)
    assert df.shape == beginshape
    assert le is not None
    assert (df.columns == columns).all()
