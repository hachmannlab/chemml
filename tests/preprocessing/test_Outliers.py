import pandas as pd

from chemml.preprocessing import Outliers

# dummy data
df = pd.DataFrame([1, 2, 3])
df[1] = [1, 1.1, 9]
df[2] = [7] * 3
df[3] = [5.432423] * 3


class TestOutliers:
    def test_mean(self):
        f = Outliers(df, m=1.3, strategy='mean')
        assert f.shape == (2, 4)
        assert f.index[1] == 1

    def test_median(self):
        f = Outliers(df, m=1., strategy='median')
        assert f.shape == (1, 4)
        assert f.index[0] == 1
