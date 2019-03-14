import numpy as np
import pandas as pd
import pytest
import os
import pkg_resources

from chemml.utils import check_object_col
from chemml.utils import isfloat
from chemml.utils import islist
from chemml.utils import istuple
from chemml.utils import isnpdot
from chemml.utils import isint
from chemml.utils import value
from chemml.utils import update_default_kwargs

@pytest.fixture()
def data_path():
    return pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'test_files'))


def test_isfloat_exception():
    assert isfloat('1') is True
    assert isfloat('a') is False


def test_islist_exception():
    assert islist('[1]') is True
    assert islist([1]) is True
    assert islist('a') is False


def test_istuple_exception():
    assert istuple('(1,)') is True
    assert istuple((1,)) is True
    assert istuple('a') is False


def test_isnpdot_exception():
    assert isnpdot('np.') is True
    with pytest.raises(ValueError):
        assert isnpdot(np.sin) is True


def test_isint_exception():
    assert isint('1') is True
    assert isint(1) is True
    assert isint('a') is False


def test_value():
    assert value('1') == 1
    assert value('np.sin') == np.sin
    assert value('type') == 'type'


def test_check_object_col_exception(data_path):
    with pytest.raises(ValueError):
        df = pd.read_csv(os.path.join(data_path, 'test_missing_values.csv'), header=None)
        f = check_object_col(df, 'df')


def test_check_object_col():
    df = pd.DataFrame()
    df[0] = ['a', 'b', 'c']
    df[1] = [1, 2, 3]
    f = check_object_col(df, 'df')

def test_update_default_kwargs():
    default_kw = {'a':4, 'b':7, 'c':8}
    kw = {'a':5}
    full_dict = update_default_kwargs(default_kw,kw)
    assert full_dict['a']==5
    full_dict = update_default_kwargs(default_kw,{})
    assert full_dict['a']==4

def test_update_default_kwargs_exception():
    default_kw = {'a': 4, 'b': 7, 'c': 8}
    kw = {'l': 5}
    with pytest.raises(ValueError):
        _ = update_default_kwargs(default_kw, kw)
    with pytest.raises(ValueError):
        _ = update_default_kwargs(default_kw, kw, 'function', 'https://...')




