import pandas as pd
import numpy as np
from ..utils.utilities import list_del_indices
import warnings

def input_preserver(learner, data, target=None):
    """
    to preserve input type, eg. transform numpy array to dataframe and keep track
    of header.
    
    Parameters
    ----------
    learner: The learner is the last fitted classifier/regressor
    data: data dataframe
    target: default None, target dataframe
    
    Returns
    -------
    transformed data and target dataframes. 
    
    """
    
    if target == None:
        #learner functions
        #return data
    else:
        #learner functions
        #return data, target
    
def Imputer_dataframe(transformer, df):
    """ keep track of features (columns) that can be removed or changed in the 
        Imputer by transforming data back to pandas dataframe structure. This happens based on
        the "statistics_" attribute of Imputer.
    
    Parameters
    ----------
    transformer: sklearn Imputer class 
         The class with adjusted parameters.
         
    df: Pandas dataframe
        The dataframe that imputer is going to deal with.
    
    Returns
    -------
    transformed data frame
    fitted imputer class
    """    
    df_columns = list(df.columns)
    df = transformer.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
        return transformer, df
    else:
        stats = transformer.statistics_
        nan_ind = [i for i,val in enumerate(stats) if np.isnan(val)] 
        df_columns = list_del_indices(df_columns, nan_ind)
        df = pd.DataFrame(df,columns=df_columns)
        return df

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
        warnings.warn("number of columns before and after transform doesn't match",Warning)
    return df

def selector_dataframe(transformer, df, tf):
    """ keep track of features (columns) that can be removed or changed in the 
        VarianceThreshold by transforming data back to pandas dataframe structure. 
        This happens based on the "get_support" method of selector.
    
    Parameters
    ----------
    transformer: sklearn VarianceThreshold class
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
        retained_features_ind = transformer.get_support(True)
        df_columns = [df_columns[i] for i in retained_features_ind]
        df = pd.DataFrame(df,columns=df_columns)
        return df
