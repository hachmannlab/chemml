import unittest
import os
import pkg_resources
import pandas as pd
import numpy as np



DATA_PATH = pkg_resources.resource_filename('cheml', os.path.join('tests', 'data'))

from cheml.preprocessing import MissingValues


# dummy data
df = pd.read_csv(os.path.join(DATA_PATH,'test_missing_values.csv'), header=None)
target=pd.DataFrame([1,2,3,np.nan,4])


class TestConstantColumns(unittest.TestCase):
    def test_zero(self):
        mv = MissingValues(strategy = 'zero',
                           string_as_null = True,
                           inf_as_null = True,
                           missing_values = None)
        f = mv.fit_transform(df)
        t = mv.fit_transform(target)
        self.assertEqual((5,9), f.shape)
        self.assertEqual(0.0, f[10][0])
        self.assertEqual(0.0, f[10][2])
        self.assertEqual(0.0, f[2][1])
        self.assertEqual(0.0, f[1][0])

if __name__== '__main__':
    unittest.main()

