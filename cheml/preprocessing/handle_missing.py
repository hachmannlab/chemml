import pandas as pd
import numpy as np
import warnings

from ..utils.validation import isfloat
from ..utils.utilities import list_del_indices
__all__ = [
    'missing_values',
]

def cut_df(col, df, paste_col=False, on_right=False):
    """ 
    To cut one or more columns from a dataframe as seprate dataframe.
    paste_col sets optional columns for the resulted dataframe. Both col and 
    paste_col must be lists.
    on_right: select as many columns as length of 'col' from right side of 
    dataframe. 
    Notice: The order must had been considered in the 'paste_col'
    """ 
    if on_right:
        n = len(col)
        df_paste = df.iloc[:,-n:]
        if paste_col:
            df_paste.columns = paste_col
        df = df.iloc[:,:-n]
    else:
        df_paste = df[col]
        if paste_col:
            df_paste.columns = paste_col
        df.drop(col,axis=1, inplace=True)
    return df, df_paste
    
def _check_object_col(df, name):
    """
    Goals: 
    - check if columns with type 'object' don't have elements that can be 
      converted to numeric values.
    - remove columns with all non numeric elements.
    """
    object_cols = [df.dtypes.index[i] for i, typ in enumerate(df.dtypes) if typ == "object"]
    for col in object_cols:
        for i, value in enumerate(df[col]):
            if  isfloat(value):
                raise ValueError("column '%s' in '%s' includes both string and float values." %(str(col),name))
    # drop object columns
    if len(object_cols)>0:
        df = df.drop(object_cols,1)
    return df
    
class missing_values(object):
    """ Handle all the missing values.
    
    Parameters
    ----------
    strategy: string, optional (default="ignore_row")
        
        list of strategies:
        - interpolate: interpolate based on sorted target values
        - zero: set to the zero
        - ignore_row: remove the entire row in data and target
        - ignore_column: remove the entire column in data and target

    string_as_null: boolean, optional (default=True)
        If True non numeric elements are considered to be null in computations.
    
    missing_values: list, optional (default=False)
        The placeholder for missing values. It must be a list of one or more of
        any type of string, float or integer values. 

    inf_as_null: boolean, optional (default=True)
        If True inf and -inf elements are considered to be null in computations.

    Returns
    -------
    data frame
    mask: Only if strategy = ignore_row. Mask is a binary pandas series which stores the information regarding removed
    """
    def __init__(self, strategy="ignore_row", string_as_null = True,
                 inf_as_null = True, missing_values = False):
        self.strategy = strategy
        self.string_as_null = string_as_null
        self.inf_as_null = inf_as_null
        self.missing_values = missing_values
        
    def fit_transform(self, df):
        """
        use fit_transform for:
            - replace missing values with nan.
            - drop columns with all nan values.
            - fill nan values with the specified strategy.

        :param:
            df: pandas data frame
        :attribute:
            mask: binary pandas series, only if strategy = 'ignore_row' or 'ignore_column'
                mask is a binary vector whose length is the number of rows/indices in the df. The index of each bit shows
                if the row/column in the same position has been removed or not.
                The goal is keeping track of removed rows/columns to change the target data frame or other input data frames based
                on that. The mask can later be used in the transform method to change other data frames in the same way.
        """
        if self.inf_as_null == True:
            df.replace([np.inf, -np.inf,'inf','-inf'], np.nan, True)
        if self.string_as_null == True:
            df = df.convert_objects(convert_numeric=True)
        if self.missing_values and isinstance(self.missing_values, (list, tuple)):
            for pattern in self.missing_values:
                df.replace(pattern, np.nan, True)

        df = _check_object_col(df, 'df')
        # drop null columns
        df.dropna(axis=1, how='all', inplace=True)

        if self.strategy == 'zero':
            for col in df.columns:
                df[col].fillna(value=0,inplace=True)
            return df
        elif self.strategy == 'ignore_row':
            dfi = df.index
            df.dropna(axis=0, how='any', inplace=True)
            mask=[i in df.index for i in dfi]
            self.mask = pd.Series(mask, index=df.index)
            # self.mask = pd.notnull(df).all(1)
            # df = df[self.mask]
            return df
        elif self.strategy == 'ignore_column':
            dfc = df.columns
            df.dropna(axis=1, how='any', inplace=True)
            mask=[i in df.columns for i in dfc]
            self.mask = pd.Series(mask, index=df.columns)
            # self.mask = pd.notnull(df).all(0)
            # df = df.T[self.mask].T
            return df
        elif self.strategy == 'interpolate':
            df = df.interpolate()
            df.fillna(method='ffill',axis=1, inplace=True) # because of nan in the first and last element of column
            return df
        else:
            msg = "Wrong strategy has been passed"
            raise TypeError(msg)

    def transform(self, df):
        """
        Only if the class is fitted with 'ignore_row' or 'ignore_column' strategies.

        :param df: pandas dataframe
        :return: transformed data frame based on the mask vector from fit_transform method.
        """
        if self.strategy == 'ignore_row':
            return df[self.mask]
        elif self.strategy == 'ignore_column':
            return df.loc[:,self.mask]
        else:
            msg = "The transform method doesn't change the dataframe if strategy='zero' or 'interpolate'. You should fit_transform your new dataframe with those methods."
            warnings.warn(msg)
