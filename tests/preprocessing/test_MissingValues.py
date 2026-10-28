import os
import pandas as pd
import numpy as np
from importlib.resources import files

from chemml.preprocessing import MissingValues

DATA_PATH = str(files('chemml').joinpath('datasets', 'data', 'test_files'))

df = pd.read_csv(os.path.join(DATA_PATH, 'test_missing_values.csv'), header=None)
target = pd.DataFrame([1, 2, 3, np.nan, 4])


class TestMissingValues:
    def test_zero(self):
        f = MissingValues(df, strategy='zero', string_as_null=True, inf_as_null=True, missing_values=None)
        assert f.shape == (5, 9)
        assert f[10][0] == 0.0
        assert f[10][2] == 0.0
        assert f[2][1] == 0.0
        assert f[1][0] == 0.0

