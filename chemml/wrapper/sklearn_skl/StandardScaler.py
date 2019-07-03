import numpy as np
import pandas as pd
import sklearn

from chemml.wrapper.database.containers import Input, Output, Parameter
from chemml.wrapper.base import BASE

class StandardScaler(BASE):
    task = 'Prepare'
    subtask = 'scaler'
    host = 'sklearn'
    function = 'StandardScaler'
    modules = ('sklearn','preprocessing')
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"

    class Inputs:
        df = Input("df","pandas dataframe", (pd.core.frame.DataFrame,))
        api = Input("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing.data.StandardScaler'>",))

    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing.data.StandardScaler'>",))

    class Parameters:
        copy = Parameter('copy', True, (bool,))
        with_mean = Parameter('with_mean', True, (bool,))
        with_std = Parameter('with_std', True, (bool,))

    class Attribures:
        scale_ = Parameter('scale_', typ=(np.ndarray, type(None)))
        mean_ = Parameter('mean_', typ=(np.ndarray, type(None)))
        var_ = Parameter('var_', typ=(np.ndarray, type(None)))
        n_samples_seen_ = Parameter('n_samples_seen_', typ=(np.ndarray, int))

    class Methods:
        method = Parameter(name="method",
                           default='fit_transform',
                           options=['fit','fit_transform','transform','inverse_transform'])

    class GUI:
        gui_meta_data = None

    def fit(self):
        pass