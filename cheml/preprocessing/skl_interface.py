import pandas as pd
import numpy as np

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

def VT_selector_dataframe(imputer, df):
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
