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
    df = scaler.fit_transform(df)
    if df.shape[1] == 0:
        warnings.warn("empty dataframe: all columns have been removed",Warning)
    df = pd.DataFrame(df,columns=df_columns)
    return scaler, df