import pandas as pd
import numpy as np
from scipy import stats

# from ..utils.validation import isfloat

__all__ = [
    'missing_values',
]

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False		

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
    method: string, optional (default="mean")
        
        list of methods:
        - mean: set to the mean
        - median: set to the median
        - most_frequent: set to the mode
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
    
    Attributes
    ----------
    
    
    
    Returns
    -------
    data and target
    """
    def __init__(self, method="mean", string_as_null = True,
                 inf_as_null = True, missing_values = False):
        self.method = method
        self.string_as_null = string_as_null
        self.inf_as_null = inf_as_null
        self.missing_values = missing_values
        
    def fit(self, data, target):
        if self.inf_as_null == True:
            data.replace([np.inf, -np.inf,'inf'], np.nan, True)
            target.replace([np.inf, -np.inf,'inf'], np.nan, True)
        if self.string_as_null == True:
            data = data.convert_objects(convert_numeric=True)
            target = target.convert_objects(convert_numeric=True)
            data = _check_object_col(data, 'data')
            target = _check_object_col(target, 'target')
        if self.missing_values and isinstance(self.missing_values, (list, tuple)):
            for pattern in self.missing_values:
                data.replace(pattern, np.nan, True)
                target.replace(pattern, np.nan, True) 
        # drop null columns
        data.dropna(axis=1, how='all', inplace=True)
        target.dropna(axis=1, how='all', inplace=True)
        
        if self.method == 'mean':
            for col in data.columns:
                mean_value = np.mean(data[col])
                data[col].fillna(value=mean_value,inplace=True)                
            for col in target.columns:
                mean_value = np.mean(target[col])
                target[col].fillna(value=mean_value,inplace=True)                
            return data, target
        elif self.method == 'median':
            for col in data.columns:
                median_value = np.median(data[col])
                data[col].fillna(value=median_value,inplace=True)                
            for col in target.columns:
                median_value = np.median(target[col])
                target[col].fillna(value=median_value,inplace=True)                
            return data, target
        elif self.method == 'most_frequent':
            for col in data.columns:
                most_frequent_value = stats.mode(data[col])[0][0]
                data[col].fillna(value=most_frequent_value,inplace=True)                
            for col in target.columns:
                most_frequent_value = stats.mode(target[col])[0][0]
                target[col].fillna(value=most_frequent_value,inplace=True)                
            return data, target
        elif self.method == 'zero':
            for col in data.columns:
                data[col].fillna(value=0,inplace=True)                
            for col in target.columns:
                target[col].fillna(value=0,inplace=True)                
            return data, target
        elif self.method == 'ignore':
            data = pd.concat([data, target], axis=1)
            data.dropna(axis=0, how='any', inplace=True)
            data, target = cut_df(list(target.columns), data, paste_col=list(target.columns), on_right=True)
            return data, target
        elif self.method == 'interpolate':
            data = pd.concat([data, target], axis=1)
            data = data.interpolate()
            data.fillna(method='ffill',axis=1, inplace=True) # because of nan in the first and last element of column
            data, target = cut_df(list(target.columns), data, paste_col=list(target.columns), on_right=True)
            return data, target