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
    object_cols = [df.dtypes.index[i] for i, type in enumerate(df.dtypes) if type == "object"]
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
    strategy: string, optional (default="mean")
        
        list of strategies:
        - interpolate: interpolate based on sorted target values
        - zero: set to the zero
        - ignore: remove the entire row in data and target
    
    string_as_null: boolean, optional (default=True)
        If True non numeric elements are considered to be null in computations.
    
    missing_values: list, optional (default=False)
        The plceholder for missing values. It must be a list of one or more of 
        any type of string, float or integer values. 

    inf_as_null: boolean, optional (default=True)
        If True inf and -inf elements are considered to be null in computations.

    Returns
    -------
    data and target
    """
    def __init__(self, strategy="interpolate", string_as_null = True,
                 inf_as_null = True, missing_values = False):
        self.strategy = strategy
        self.string_as_null = string_as_null
        self.inf_as_null = inf_as_null
        self.missing_values = missing_values
        
    def fit(self, df):
        """
        fit changes all missing values to nan. Then, they would be ready to be 
        filled with pandas.fillna or sklearn.Imputer with specific strategies. 
        """
        if self.inf_as_null == True:
            df.replace([np.inf, -np.inf,'inf','-inf'], np.nan, True)
        if self.string_as_null == True:
            df = df.convert_objects(convert_numeric=True)
        if self.missing_values and isinstance(self.missing_values, (list, tuple)):
            for pattern in self.missing_values:
                df.replace(pattern, np.nan, True)
        return df
        
    def transform(self, data, target):    
        data = _check_object_col(data, 'data')
        target = _check_object_col(target, 'target')
        # drop null columns
        data.dropna(axis=1, how='all', inplace=True)
        target.dropna(axis=1, how='all', inplace=True)
        
        if self.strategy == 'zero':
            for col in data.columns:
                data[col].fillna(value=0,inplace=True)                
            for col in target.columns:
                target[col].fillna(value=0,inplace=True)                
            return data, target
        elif self.strategy == 'ignore':
            data = pd.concat([data, target], axis=1)
            data.dropna(axis=0, how='any', inplace=True)
            data, target = cut_df(list(target.columns), data, paste_col=list(target.columns), on_right=True)
            return data, target
        elif self.strategy == 'interpolate':
            data = pd.concat([data, target], axis=1)
            data = data.interpolate()
            data.fillna(method='ffill',axis=1, inplace=True) # because of nan in the first and last element of column
            data, target = cut_df(list(target.columns), data, paste_col=list(target.columns), on_right=True)
            return data, target

def Imputer_dataframe(imputer, df):
    """ keep track of features (columns) that can be removed or changed in the 
        Imputer by transforming data back to pandas dataframe structure. This happens based on
        the "statistics_" attribute of Imputer.
    
    Parameters
    ----------
    imputer: sklearn Imputer class 
         The class with adjusted parameters.
         
    df: Pandas dataframe
        The dataframe that imputer is going to deal with.
    
    Returns
    -------
    transformed data frame
    fitted imputer class
    """    
    df_columns = list(df.columns)
    df = imputer.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
    stats = imputer.statistics_
    nan_ind = [i for i,val in enumerate(stats) if np.isnan(val)] 
        df_columns = list_del_indices(df_columns, nan_ind)
    df = pd.DataFrame(df,columns=df_columns)
    return imputer, df
