import numpy as np
import pytest
import time

from chemml.utils import list_del_indices
from chemml.utils import std_datetime_str
from chemml.utils import tot_exec_time_str
from chemml.utils import chunk
from chemml.utils import bool_formatter


def test_list_del_indices():
    mylist = list_del_indices([9,3,5,7,1], [4,2])
    assert len(mylist) == 3
    assert mylist == [9, 3, 7]


def test_std_datetime_str():
    s = std_datetime_str(mode = 'datetime')
    assert s[-3] == ':'
    s = std_datetime_str(mode = 'date')
    assert s[-3] == '-'
    s = std_datetime_str(mode = 'time')
    assert s[-3] == ':'
    s = std_datetime_str(mode = 'datetime_ms')
    assert s[4] == '-'
    s = std_datetime_str(mode = 'time_ms')
    assert s[2] == ':'
    with pytest.raises(ValueError):
        std_datetime_str(mode='hour')


def test_exec_time_str():
    time_start = time.time()
    time.sleep(0.5)
    s = tot_exec_time_str(time_start)
    assert int(s[-4]) >= 5


def test_chunk():
    x = np.array(range(10))
    y = np.array(range(20,30))
    it = chunk(range(len(x)), 3, x, y)
    x_chunk, y_chunk = next(it)
    assert len(x_chunk) == 4


def test_bool_formatter():
    bf_true = bool_formatter(True)
    bf_false = bool_formatter(False)
    assert bf_true == 'true'
    assert bf_false == 'false'


def test_bool_formatter_exception():
    with pytest.raises(ValueError):
        bool_formatter('true')
