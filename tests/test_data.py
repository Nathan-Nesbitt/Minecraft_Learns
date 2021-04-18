import pytest

from pandas import DataFrame

from minecraft_learns.data import Data
from minecraft_learns.errors import NoDataStored

def test_load_data():
    """
    Checks failure of non acceptable type is input
    """
    try:
        Data("test.txt").load_data()
    except FileNotFoundError:
        assert True
    except NoDataStored:
        assert True
    else:
        assert False
