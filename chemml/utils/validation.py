import numpy as np
import pandas as pd
import copy

# Todo: check_object_col is really inefficient (iteration on the values of each column)


def isfloat(val):
    """
    check if the entry can become a float (float or string of float)

    Parameters
    ----------
    val
        an entry of any type

    Returns
    -------
    bool
        True if the input can become a float, False otherwise

    """
    try:
        float(val)
        return True
    except ValueError:
        return False


def islist(val):
    """
    check if the entry is a list or is a string of list

    Parameters
    ----------
    val
        an entry of any type

    Returns
    -------
    bool
        True if the input is either a list or a string of list, False otherwise

    """
    text = str(val)
    if text[0] == '[' and text[len(text) - 1] == ']':
        return True
    else:
        return False


def istuple(val):
    """
    check if the entry is a tuple or is a string of tuple

    Parameters
    ----------
    val
        an entry of any type

    Returns
    -------
    bool
        True if the input is either a tuple or a string of tuple, False otherwise

    Notes
    -----
    please note that '(1)' is also a tuple, while (1) is not a tuple

    """
    text = str(val)
    if text[0] == '(' and text[len(text) - 1] == ')':
        return True
    else:
        return False


def isnpdot(val):
    """
    check if the entry starts with 'np.'

    Parameters
    ----------
    val: str
        a string entry

    Returns
    -------
    bool
        True if the entry is a string and the first three characters are 'np.',
        False otherwise

    """
    if isinstance(val, str):
        if val[0:3] == 'np.':
            return True
        else:
            return False
    else:
        msg = 'entry must be string'
        raise ValueError(msg)


def isint(val):
    """
    check if the entry can become an integer (integer or string of integer)

    Parameters
    ----------
    val
        an entry of any type

    Returns
    -------
    bool
        True if the input can become an integer, False otherwise

    """
    # print("hello",val)
    try:
        int(val)
        return True
    except ValueError:
        return False


def value(entry):
    """
    check if the string entry can be evaluated to python data structure

    Parameters
    ----------
    entry: str
        an entry of type str

    Returns
    -------
    bool
        True if the input can become an integer, False otherwise

    Notes
    -----
    returns 'type' for the entry 'type', although type is a code object

    """
    # Todo: try to replace bare except
    try:
        val = eval(entry)
        if isinstance(val, type):
            return entry
        else:
            return val
    except:
        return entry


# def check_input(X,name,n0=None,n1=None, format_out='df'):
#     """
#     Makes sure that input is a 2-D numpy array or pandas dataframe in the correct format.
#
#     :param X: numpy.ndarray or pandas.DataFrame
#         input data
#     :param name: string
#         name of input (e.g. training input)
#     :param n0: int
#         number of data entries
#     :param n1: int
#         number of features
#     :param format_out: string ('df' or 'ar'), optional (default = 'df')
#
#     :return input data converted to array or dataframe
#     :return the header of dataframe
#         if input data is not a dataframe return None
#
#     """
#     if not (X.ndim == 2):
#         raise Exception(name+' needs to be two dimensional')
#     if isinstance(X, pd.DataFrame):
#         if format_out == 'ar':
#             if X.shape[1]>1:
#                 header = X.columns
#                 X = X.values
#             else:
#                 if n0 == 1:
#                     header = X.columns
#                     X = X[header[0]].values
#                 else:
#                     header = X.columns
#                     X = X.values
#         else:
#             header = X.columns
#         if not np.can_cast(X.dtypes, np.float, casting='same_kind'):
#             raise Exception(name + ' cannot be cast to floats')
#     elif isinstance(X, np.ndarray):
#         if format_out == 'df':
#             X = pd.DataFrame(X)
#             header = None
#         else:
#             header = None
#     else:
#         raise Exception(name+' needs to be either pandas dataframe or numpy array')
#     if n0 and X.shape[0] != n0:
#         raise Exception(name+' has an invalid number of data entries')
#     if n1 and X.shape[1] != n1:
#         raise Exception(name+' has an invalid number of feature entries')
#     return X.astype(float), header


def check_object_col(df, name):
    """
    check if columns with type 'object' don't have elements that can be
      converted to numeric values.
    remove columns with all non numeric elements.

    Parameters
    ----------
    df: pandas dataframe
        input dataframe
    name: str
        variable name of the dataframe for internal usage, e.g. error message handling

    Returns
    -------
    pandas dataframe
        modified dataframe
    """
    object_cols = [df.dtypes.index[i] for i, typ in enumerate(df.dtypes) if typ == "object"]
    for col in object_cols:
        for i, value in enumerate(df[col]):
            if isfloat(value):
                raise ValueError("column '%s' in '%s' includes both string and float values." %(str(col), name))
    # drop object columns
    if len(object_cols) > 0:
        df = df.drop(object_cols, 1)
    return df

def update_default_kwargs(default_kw, kw, method_name=None, method_doc_path=None):
    """
    This function receives a spuer-dictionary (e.g., default kwargs) and update the values with a sub-dictionary (e.g., kwargs).

    Parameters
    ----------
    default_kw: dict
        The default dictionary

    kw: dict
        The (partially) updated dictionary values. The keys must be in the default_kw.

    Raises
    ------
    ValueError
        If the keys in the default_kw is not a superset of the keys inside the kw.

    Returns
    -------
    dict
        The full default dictionary with all the keys, but potentially updated values based on the second dicitonary.

    """
    temp = copy.deepcopy(default_kw)
    for k in kw:
        if k not in temp:
            if method_name:
                msg = "The argument '%s' is not a valid parameter for the method '%s'.\n" \
                      "You can check the documentation page for a complete list of legit arguments: %s" %(k, method_name, str(method_doc_path))
            else:
                msg = "The argument '%s' is not a valid parameter for the method.\n" \
                      "You can check the documentation page for a complete list of legit arguments." %(k)
            raise ValueError(msg)
        else:
            temp[k] = kw[k]
    return temp

