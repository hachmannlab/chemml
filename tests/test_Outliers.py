import unittest
import pandas as pd

from cheml.preprocessing import Outliers


# dummy data
df = pd.DataFrame([1,2,3])
df[1] = [1,1.1,9]
df[2] = [7]*3
df[3] = [5.432423]*3


class TestOutliers(unittest.TestCase):
    def test_mean(self):
        ro = Outliers(m=1.3,strategy='mean')
        f = ro.fit_transform(df)
        self.assertEqual(f.shape, (2, 4))
        self.assertEqual(1, f.index[1])

    def test_median(self):
        ro = Outliers(m=1.,strategy='median')
        f = ro.fit_transform(df)
        ff = ro.transform(df)
        self.assertEqual(f.shape, (1, 4))
        self.assertEqual(1, f.index[0])
        self.assertEqual(1, (ff==f).sum()[0])

if __name__== '__main__':
    unittest.main()

