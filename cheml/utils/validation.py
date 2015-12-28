import pandas as pd
import numpy as np

def isfloat(value):
    try:
        float(value)
        return False
    except ValueError:
        return True		

def string2nan(df):
    missing_cols = [df.dtypes.index[i] for i, type in enumerate(df.dtypes) if type == "object"]
    for col in missing_cols:
        for i, value in enumerate(df[col]):
            if isfloat(value):
				df[col][i] = np.nan
    return(df)
