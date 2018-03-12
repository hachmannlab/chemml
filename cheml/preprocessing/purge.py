import numpy as np

class ConstantColumns(object):
    """
    remove constant columns


    Attributes
    ----------
    removed_columns: list of column headers that have been removed

    Returns
    -------
    df: pandas dataframe

    """
    def fit_transform(self,df):
        """
        fit the method to the input dataframe and change it accordingly

        Parameters
        ----------
        df: pandas dataframe
            input dataframe

        Returns
        -------
        transformed dataframe
        """
        dfc = df.columns
        df = df.loc[:, (df != df.ix[0]).any()]
        self.removed_columns_ = np.array([i for i in dfc if i not in df.columns])
        return df
    def transform(self,df):
        """
        find and remove headers that are in the removed_columns_ attribute of the previous fit_transform method

        Parameters
        ----------
        df: pandas dataframe
            input dataframe

        Returns
        -------
        transformed dataframe
        """
        df = df.drop(self.removed_columns_,1)
        return df

class Outliers(object):
    """
    remove all rows where the values of a certain column are within an specified
    standard deviation from mean/median.

    Parameters
    ----------
    m: float, optional (default=3.0)
        the outlier threshold with respect to the standard deviation

    strategy: string, optional (default='median')
        available options: 'mean' and 'median'
        Values of each column will be compared to the 'mean' or 'median' of that column.

    Attributes
    ----------
    removed_rows_: numpy array of indices that have been removed

    Notes
    -----
    We highly recommend you to remove constant columns first and then remove outliers.
    """
    def __init__(self, m=2.0, strategy = 'median'):
        self.m = m
        self.strategy = strategy
    def fit_transform(self,df):
        """
        fit the method to the input dataframe and change it accordingly

        Parameters
        ----------
        df: pandas dataframe
            input dataframe

        Returns
        -------
        transformed dataframe
        """
        if self.strategy == 'mean':
            mask = ((df - df.mean()).abs() <= self.m * df.std(ddof=0)).T.all()
        elif self.strategy == 'median':
            mask = (((df - df.median()).abs()) <= self.m * df.std(ddof=0)).T.all()
        df = df.loc[mask, :]
        self.removed_rows_ = np.array(mask[mask==False].index)
        return df
    def transform(self,df):
        """
        find and remove rows/indices that are in the removed_rows_ attribute of the previous fit_transform method

        Parameters
        ----------
        df: pandas dataframe
            input dataframe

        Returns
        -------
        transformed dataframe
        """
        df = df.drop(self.removed_rows_,0)
        return df
