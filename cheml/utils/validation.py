import numpy as np

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
        return eval(string)
    except NameError:
        return string