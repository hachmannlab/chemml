import pandas as pd
import numpy as np

class Constant(object):
    """
    remove constant columns

    :param

    :arg
        removed_columns: list of column headers that have been removed

    :return
        df: pandas dataframe

    """
    def fit_transform(self,df):
        dfc = df.columns
        df = df.loc[:, (df != df.ix[0]).any()]
        self.removed_columns_ = np.array([i for i in dfc if i not in df.columns])
        return df
    def transform(self,df):
        df = df.drop(self.removed_columns_,1)
        return df

class Remove_Outliers(object):
    """
    remove all rows where the values of a certain column are within an specified
    standard deviation from mean/median.

    Note that this method first removes all columns with constant values.

    :param:
    m: integer or float - optional (default=3)
        the outlier threshold with respect to the standard deviation

    strategy: string = optional (default='median')
        available options: 'mean' and 'median'
        Values of each column will be compared to the 'mean' or 'median' of that column.
    """
    def __init__(self, m=2, strategy = 'median'):
        self.m = m
        self.strategy = strategy
    def fit_transform(self,df):
        api = Constant()
        df = api.fit_transform(df)
        self.removed_columns_ = api.removed_columns_
        if self.strategy == 'mean':
            mask = ((df - df.mean()).abs() < self.m * df.std(ddof=0)).T.all()
        elif self.strategy == 'median':
            mask = (((df - df.median()).abs()) < self.m * df.std(ddof=0)).T.all()
        df = df.loc[mask, :]
        self.removed_rows_ = np.array(mask[mask==False].index)
        return df

class ManytoMany(object):
    """
    remove constant features

    :param
    fraction: float, optional(default=1.0)
    float in the interval [0.0,1.0]. It is the fraction of total number of data points
    with identical feature value.


    :return

    """

    def __init__(self, fraction=1.0):
        self.fraction = fraction

    def fit_transform(self, df):
        df_out = df.loc[:, (df != df.ix[0]).any()]
        return df_out

        purged_features = []
        for i in df.columns:
            max_num_des = list(df[i].value_counts())[0]
            if max_num_des > f_remove * max_size_class:
                df = df.drop(i, 1)
                count += 1
                remove_list.append(i)
