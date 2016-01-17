import pandas as pd
import numpy as np
from ..utils.utilities import list_del_indices

def transformer_dataframe(transformer, df):
    """ keep track of features (columns) that can be removed or changed in the 
        Scaler by transforming data back to pandas dataframe structure. 
    
    Parameters
    ----------
    scaler: sklearn Scaler class 
         The class with adjusted parameters.
         
    df: Pandas dataframe
        The dataframe that scaler is going to deal with.
    
    Returns
    -------
    transformed data frame
    fitted scaler class

    """
    df_columns = list(df.columns)
    df = transformer.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
    if df.shape[1] == len(df_columns):
        df = pd.DataFrame(df,columns=df_columns)
    else:
        warnings.warn("number of columns befor and after transform doesn't match",Warning)
    return scaler, df

def selector_dataframe(transformer, df, tf):
    """ keep track of features (columns) that can be removed or changed in the 
        VarianceThreshold by transforming data back to pandas dataframe structure. 
        This happens based on the "get_support" method of Imputer.
    
    Parameters
    ----------
    imputer: sklearn VarianceThreshold class 
         The class with adjusted parameters.
         
    df: Pandas dataframe
        data frame
    
    tf: Pandas dataframe
        target frame 
    
    Returns
    -------
    transformed data frame
    fitted imputer class
    """    
    df_columns = list(df.columns)
    df = transformer.fit_transform(df,tf)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
        return transformer, df
    else:
        retained_features_ind = sel.get_support(True)
        df_columns = [df_columns[i] for i in retained_features_ind]
        df = pd.DataFrame(df,columns=df_columns)
        return transformer, df
