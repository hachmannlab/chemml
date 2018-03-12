import unittest
import pandas as pd

from cheml.preprocessing import ConstantColumns


# dummy data
df = pd.DataFrame([1,2,3])
df[1] = ['a']*3
df[2] = [7]*3
df[3] = [5.432423]*3


class TestConstantColumns(unittest.TestCase):
    def test_fit_transform(self):
        cc = ConstantColumns()
        f = cc.fit_transform(df)
        self.assertEqual(f.shape, (3, 1))
        self.assertEqual(0, f.columns[0])

    def test_transform(self):
        cc = ConstantColumns()
        f = cc.fit_transform(df)
        ff = cc.transform(df)
        self.assertEqual(3, (ff==f).sum()[0])

if __name__== '__main__':
    unittest.main()

