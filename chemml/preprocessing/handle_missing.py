import warnings
from builtins import range

import pandas as pd
import numpy as np

from chemml.utils import check_object_col


def MissingValues(df, strategy="ignore_row",
                    string_as_null=True,
                    inf_as_null=True,
                    missing_values=None):
    """
    find missing values and interpolate/replace or remove them.

    Parameters
    ----------
    df : pandas dataframe

    strategy: string, optional (default="ignore_row")
        
        list of strategies:
        - interpolate: interpolate based on sorted target values
        - zero: set to the zero
        - ignore_row: remove the entire row in data and target
        - ignore_column: remove the entire column in data and target

    string_as_null: boolean, optional (default=True)
        If True non numeric elements are considered to be null in computations.
    
    missing_values: list, optional (default=None)
        where you define specific formats of missing values. It is a list of string, float or integer values.

    inf_as_null: boolean, optional (default=True)
        If True inf and -inf elements are considered to be null in computations.

    Returns
    -------
    dataframe

    Notes
    ----------
    mask is a binary vector whose length is the number of rows/indices in the df. The index of each bit shows
    if the row/column in the same position has been removed or not.
    The goal is keeping track of removed rows/columns to change the target data frame or other input data frames based
    on that. The mask can later be used in the transform method to change other data frames in the same way.
    """
    if inf_as_null == True:
        df.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan, True)
    if string_as_null == True:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if isinstance(missing_values, (list, tuple)):
        for pattern in missing_values:
            df.replace(pattern, np.nan, True)

    df = check_object_col(df, 'df')
    # drop null columns
    df = df.dropna(axis=1, how='all', inplace=False)

    if strategy == 'zero':
        for col in df.columns:
            df[col].fillna(value=0, inplace=True)
        return df
    elif strategy == 'ignore_row':
        dfi = df.index
        df = df.dropna(axis=0, how='any', inplace=False)
        mask = [i in df.index for i in dfi]
        mask = pd.Series(mask, index=dfi)
        # mask = pd.notnull(df).all(1)
        # df = df[mask]
        return df
    elif strategy == 'ignore_column':
        dfc = df.columns
        df = df.dropna(axis=1, how='any', inplace=False)
        mask = [i in df.columns for i in dfc]
        mask = pd.Series(mask, index=dfc)
        # mask = pd.notnull(df).all(0)
        # df = df.T[mask].T
        return df
    elif strategy == 'interpolate':
        df = df.interpolate()
        df = df.fillna(
            method='ffill', axis=1, inplace=False
        )  # because of nan in the first and last element of column
        return df
    else:
        msg = "Wrong strategy has been passed"
        raise TypeError(msg)

def Outliers(df, m=2.0,strategy='median'):
    """
    remove all rows where the values of a certain column are within an specified
    standard deviation from mean/median.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    m: float, optional (default=3.0)
        the outlier threshold with respect to the standard deviation

    strategy: string, optional (default='median')
        available options: 'mean' and 'median'
        Values of each column will be compared to the 'mean' or 'median' of that column.
    
    Returns
    -------
    dataframe

    Notes
    -----
    We highly recommend you to remove constant columns first and then remove outliers. 

    """
    if strategy == 'mean':
        mask = ((df - df.mean()).abs() <= m * df.std(ddof=0)).T.all()
    elif strategy == 'median':
        mask = (((df - df.median()).abs()) <=
                m * df.std(ddof=0)).T.all()
    df = df.loc[mask, :]
    removed_rows_ = np.array(mask[mask == False].index)
    return df

def ConstantColumns(df):
    """
    remove constant columns

    Parameters
    ----------
    df: pandas dataframe
        input dataframe

    Returns
    -------
    df: pandas dataframe

    """
    dfc = df.columns
    df = df.loc[:, (df != df.iloc[0]).any()]
    removed_columns_ = np.array(
        [i for i in dfc if i not in df.columns])
    return df

# def transform_constant_cols(self, df):
#     """
#     find and remove headers that are in the removed_columns_ attribute of the previous fit_transform method

#     Parameters
#     ----------
#     df: pandas dataframe
#         input dataframe

#     Returns
#     -------
#     transformed dataframe
#     """
#     df = df.drop(self.removed_columns_, 1)
#     return df
# def transform_outliers(self, df):
#     """
#     find and remove rows/indices that are in the removed_rows_ attribute of the previous fit_transform method

#     Parameters
#     ----------
#     df: pandas dataframe
#         input dataframe

#     Returns
#     -------
#     transformed dataframe
#     """
#     df = df.drop(removed_rows_, 0)
#     return df

# def transform(self, df):
#     """
#     Only if the class is fitted with 'ignore_row' or 'ignore_column' strategies.

#     Parameters
#     ----------
#     df : pandas dataframe

#     Returns
#     -------
#     transformed data frame based on the mask vector from fit_transform method.
#     """
#     if strategy == 'ignore_row':
#         return df[mask]
#     elif strategy == 'ignore_column':
#         return df.loc[:, mask]
#     else:
#         msg = "The transform method doesn't change the dataframe if strategy='zero' or 'interpolate'. You should fit_transform the new dataframe with those methods."
#         warnings.warn(msg)
