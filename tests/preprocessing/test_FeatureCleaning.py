import numpy as np
import pandas as pd
import pytest

from chemml.preprocessing import RemoveCorrFeatures, RemoveInvFeatures


@pytest.fixture
def inv_df():
    return pd.DataFrame({
        'a': [0, 1] * 15,
        'b': [1] * 30,
        'c': [0] * 29 + [1],
        'd': [5] * 28 + [2, 3],
        'e': np.linspace(0.0, 0.005, 30),
        'f': np.arange(30)
    })


def test_remove_correlated_columns():
    df = pd.DataFrame({
        'A': [0, 1, 2, 3, 4, 5],
        'B': [0, 1, 2, 3, 4, 5],
        'C': [0, 1, 0, 1, 0, 1],
        'D': [0, -1, -2, -3, -4, -5]
    })

    f = RemoveCorrFeatures(df, correlation_threshold=0.9)

    assert ['A', 'C'] == f.columns.tolist()


def test_invalid_threshold():
    df = pd.DataFrame({'A': [1, 2], 'B': [1, 2]})
    with pytest.raises(ValueError):
        RemoveCorrFeatures(df, correlation_threshold=0)


def test_remove_invariant_with_removed_columns(inv_df):
    cleaned, removed = RemoveInvFeatures(
        inv_df,
        sanitize_nonbinary=True,
        use_variance_filtering=True,
        variance_threshold=0.001,
        keep_filtered_columns=True
    )

    assert ['a', 'f'] == cleaned.columns.tolist()
    assert ['b', 'c', 'd', 'e'] == removed.columns.tolist()


def test_skip_nonbinary_sanitization(inv_df):
    cleaned = RemoveInvFeatures(
        inv_df,
        sanitize_nonbinary=False,
        use_variance_filtering=False,
        variance_threshold=0.001,
        keep_filtered_columns=False
    )

    assert ['a', 'd', 'e', 'f'] == cleaned.columns.tolist()
