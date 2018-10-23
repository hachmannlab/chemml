import pytest
import warnings

from chemml.initialization import Split
from chemml.datasets import load_organic_density


@pytest.fixture()
def data():
    _, _, X = load_organic_density()
    return X


def test_df_exception():
    with pytest.raises(ValueError):
        cls = Split(selection=2)
        cls.fit('df')


def test_select_exception(data):
    with pytest.raises(ValueError):
        cls = Split(selection='2')
        cls.fit(data)


def test_select_int(data):
    cls = Split(selection=2)
    x1, x2 = cls.fit(data)
    assert list(x1.columns) == ['MW', 'AMW']
    assert len(x2.columns) == 198


def test_select_int(data):
    cls = Split(selection=['MW', 'AMW'])
    x1, x2 = cls.fit(data)
    assert list(x1.columns) == ['MW', 'AMW']
    assert len(x2.columns) == 198


def test_select_warning(data):
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        cls = Split(selection=201)
        x1, x2 = cls.fit(data)
        assert len(x1.columns) == 200
        assert x2 is None
