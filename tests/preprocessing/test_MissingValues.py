import unittest
import os
import pandas as pd
import numpy as np
import pkg_resources

from chemml.preprocessing import MissingValues

DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data','test_files'))

df = pd.read_csv(os.path.join(DATA_PATH, 'test_missing_values.csv'), header=None)
target = pd.DataFrame([1, 2, 3, np.nan, 4])


class TestConstantColumns(unittest.TestCase):
    def test_zero(self):
        mv = MissingValues(strategy='zero', string_as_null=True, inf_as_null=True, missing_values=None)
        f = mv.fit_transform(df)
        t = mv.fit_transform(target)
        self.assertEqual((5, 9), f.shape)
        self.assertEqual(0.0, f[10][0])
        self.assertEqual(0.0, f[10][2])
        self.assertEqual(0.0, f[2][1])
        self.assertEqual(0.0, f[1][0])

