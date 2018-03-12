import numpy as np
import pandas as pd

def string2nan(df):
    missing_cols = [df.dtypes.index[i] for i, type in enumerate(df.dtypes) if type == "object"]
    for col in missing_cols:
        for i, value in enumerate(df[col]):
            if isfloat(value):
                df[col][i] = np.nan
    return(df)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def islist(value):
    text = str(value)
    if text[0] == '[' and text[len(text) - 1] == ']':
        return True
    else:
        return False


def istuple(value):
    text = str(value)
    if text[0] == '(' and text[len(text) - 1] == ')':
        return True
    else:
        return False


def isnpdot(value):
    text = str(value)
    if text[0:3] == 'np.':
        return True
    else:
        return False


def isint(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def value(string):
    try:
        val =  eval(string)
        if type(val)==type:
            return string
        else:
            return val
    except:
        return string

def check_input(X,name,n0=None,n1=None, format_out='df'):
    """
    Makes sure that input is a 2-D numpy array or pandas dataframe in the correct format.

    :param X: numpy.ndarray or pandas.DataFrame
        input data
    :param name: string
        name of input (e.g. training input)
    :param n0: int
        number of data entries
    :param n1: int
        number of features
    :param format_out: string ('df' or 'ar'), optional (default = 'df')

    :return input data converted to array or dataframe
    :return the header of dataframe
        if input data is not a dataframe return None

    """
    if not (X.ndim == 2):
        raise Exception(name+' needs to be two dimensional')
    if isinstance(X, pd.DataFrame):
        if format_out == 'ar':
            if X.shape[1]>1:
                header = X.columns
                X = X.values
            else:
                if n0 == 1:
                    header = X.columns
                    X = X[header[0]].values
                else:
                    header = X.columns
                    X = X.values
        else:
            header = X.columns
        if not np.can_cast(X.dtypes, np.float, casting='same_kind'):
            raise Exception(name + ' cannot be cast to floats')
    elif isinstance(X, np.ndarray):
        if format_out == 'df':
            X = pd.DataFrame(X)
            header = None
        else:
            header = None
    else:
        raise Exception(name+' needs to be either pandas dataframe or numpy array')
    if n0 and X.shape[0] != n0:
        raise Exception(name+' has an invalid number of data entries')
    if n1 and X.shape[1] != n1:
        raise Exception(name+' has an invalid number of feature entries')
    return X.astype(float), header

def check_object_col(df, name):
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
