import numpy as np
from .containers import Input, Output, Parameter, req, regression_types, cv_types

class Binarizer(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'Binarizer'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's Binarizer class", ("<class 'sklearn.preprocessing.data.Binarizer'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's Binarizer class", ("<class 'sklearn.preprocessing.data.Binarizer'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class PolynomialFeatures(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'PolynomialFeatures'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's PolynomialFeatures class", ("<class 'sklearn.preprocessing.data.PolynomialFeatures'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's PolynomialFeatures class", ("<class 'sklearn.preprocessing.data.PolynomialFeatures'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class OneHotEncoder(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'OneHotEncoder'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's OneHotEncoder class", ("<class 'sklearn.preprocessing.data.OneHotEncoder'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's OneHotEncoder class", ("<class 'sklearn.preprocessing.data.OneHotEncoder'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass

class Imputer(object):
    task = 'Prepare'
    subtask = 'preprocessor'
    host = 'sklearn'
    function = 'Imputer'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's Imputer class", ("<class 'sklearn.preprocessing.imputation.Imputer'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's Imputer class", ("<class 'sklearn.preprocessing.imputation.Imputer'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass

class KernelPCA(object):
    task = 'Prepare'
    subtask = 'feature transformation'
    host = 'sklearn'
    function = 'KernelPCA'
    modules = ('sklearn','decomposition')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's KernelPCA class", ("<class 'sklearn.decomposition.kernel_pca.KernelPCA'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's KernelPCA class", ("<class 'sklearn.decomposition.kernel_pca.KernelPCA'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', False,'Boolean',
                        description = "Always False, the header of input dataframe is not equivalent with the transformed dataframe",
                        options = (False))
    class FParameters:
        pass
class PCA(object):
    task = 'Prepare'
    subtask = 'feature transformation'
    host = 'sklearn'
    function = 'PCA'
    modules = ('sklearn','decomposition')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's PCA class", ("<class 'sklearn.decomposition.pca.PCA'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's PCA class", ("<class 'sklearn.decomposition.pca.PCA'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', False,'Boolean',
                        description = "Always False, the header of input dataframe is not equivalent with the transformed dataframe",
                        options = (False))
    class FParameters:
        pass

class Normalizer(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'Normalizer'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's Normalizer class", ("<class 'sklearn.preprocessing.data.Normalizer'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's Normalizer class", ("<class 'sklearn.preprocessing.data.Normalizer'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api None: only make a new api ",
                        options = ('fit_transform', 'transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class StandardScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'StandardScaler'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing.data.StandardScaler'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing.data.StandardScaler'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class MinMaxScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'MinMaxScaler'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's MinMaxScaler class", ("<class 'sklearn.preprocessing.data.MinMaxScaler'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's MinMaxScaler class", ("<class 'sklearn.preprocessing.data.MinMaxScaler'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class MaxAbsScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'MaxAbsScaler'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's MaxAbsScaler class", ("<class 'sklearn.preprocessing.data.MaxAbsScaler'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's MaxAbsScaler class", ("<class 'sklearn.preprocessing.data.MaxAbsScaler'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class RobustScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'RobustScaler'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's RobustScaler class", ("<class 'sklearn.preprocessing.data.RobustScaler'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's RobustScaler class", ("<class 'sklearn.preprocessing.data.RobustScaler'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass

class ShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'ShuffleSplit'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
        api = Output("api","instance of scikit-learn's ShuffleSplit class", ("<class 'sklearn.model_selection._split.ShuffleSplit'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        pass
class StratifiedShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'StratifiedShuffleSplit'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
        api = Output("api","instance of scikit-learn's StratifiedShuffleSplit class", ("<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        pass
class train_test_split(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'train_test_split'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        dfx_test = Output("dfx_test","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy_train = Output("dfy_train","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy_test = Output("dfy_test","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx_train = Output("dfx_train","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class KFold(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'KFold'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's KFold class", ("<class 'sklearn.model_selection._split.KFold'>",))
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        pass

class ARDRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'ARDRegression'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's ARDRegression class", ("<class 'sklearn.linear_model.bayes.ARDRegression'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's ARDRegression class", ("<class 'sklearn.linear_model.bayes.ARDRegression'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class BayesianRidge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'BayesianRidge'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's BayesianRidge class", ("<class 'sklearn.linear_model.bayes.BayesianRidge'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's BayesianRidge class", ("<class 'sklearn.linear_model.bayes.BayesianRidge'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class ElasticNet(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'ElasticNet'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's ElasticNet class", ("<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's ElasticNet class", ("<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class KernelRidge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'KernelRidge'
    modules = ('sklearn','kernel_ridge')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's KernelRidge class", ("<class 'sklearn.kernel_ridge.KernelRidge'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's KernelRidge class", ("<class 'sklearn.kernel_ridge.KernelRidge'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class Lars(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Lars'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's Lars class", ("<class 'sklearn.linear_model.least_angle.Lars'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's Lars class", ("<class 'sklearn.linear_model.least_angle.Lars'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class Lasso(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Lasso'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's Lasso class", ("<class 'sklearn.linear_model.coordinate_descent.Lasso'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's Lasso class", ("<class 'sklearn.linear_model.coordinate_descent.Lasso'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class LassoLars(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LassoLars'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's LassoLars class", ("<class 'sklearn.linear_model.least_angle.LassoLars'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's LassoLars class", ("<class 'sklearn.linear_model.least_angle.LassoLars'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class LinearRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LinearRegression'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's LinearRegression class", ("<class 'sklearn.linear_model.base.LinearRegression'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's LinearRegression class", ("<class 'sklearn.linear_model.base.LinearRegression'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class LinearSVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LinearSVR'
    modules = ('sklearn','svm')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's LinearSVR class", ("<class 'sklearn.svm.classes.LinearSVR'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's LinearSVR class", ("<class 'sklearn.svm.classes.LinearSVR'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class LogisticRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LogisticRegression'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's LogisticRegression class", ("<class 'sklearn.linear_model.logistic.LogisticRegression'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's LogisticRegression class", ("<class 'sklearn.linear_model.logistic.LogisticRegression'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class MLPRegressor(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MLPRegressor'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor"

    class Inputs:
        api = Input("api","instance of scikit-learn's MLPRegressor class", ("<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's MLPRegressor class", ("<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class MultiTaskElasticNet(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MultiTaskElasticNet'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's MultiTaskElasticNet class", ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's MultiTaskElasticNet class", ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class MultiTaskLasso(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MultiTaskLasso'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's MultiTaskLasso class", ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's MultiTaskLasso class", ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class NuSVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'NuSVR'
    modules = ('sklearn','svm')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's NuSVR class", ("<class 'sklearn.svm.classes.NuSVR'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's NuSVR class", ("<class 'sklearn.svm.classes.NuSVR'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class Ridge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Ridge'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's Ridge class", ("<class 'sklearn.linear_model.ridge.Ridge'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's Ridge class", ("<class 'sklearn.linear_model.ridge.Ridge'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class SGDRegressor(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'SGDRegressor'
    modules = ('sklearn','linear_model')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's SGDRegressor class", ("<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's SGDRegressor class", ("<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class SVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'SVR'
    modules = ('sklearn','svm')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"

    class Inputs:
        api = Input("api","instance of scikit-learn's SVR class", ("<class 'sklearn.svm.classes.SVR'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's SVR class", ("<class 'sklearn.svm.classes.SVR'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass

class GridSearchCV(object):
    task = 'Search'
    subtask = 'grid'
    host = 'sklearn'
    function = 'GridSearchCV'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", tuple(["<type 'str'>"]+list(regression_types())))
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<type 'str'>","<class 'sklearn.metrics.scorer._PredictScorer'>",))
    class Outputs:
        cv_results_ = Output("cv_results_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's GridSearchCV class", ("<class 'sklearn.grid_search.GridSearchCV'>",))
        best_estimator_ = Output("best_estimator_","instance of a machine learning class", regression_types())
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class cross_val_predict(object):
    task = 'Search'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_predict'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", ("<class 'sklearn.linear_model.base.LinearRegression'>","<class 'sklearn.linear_model.ridge.Ridge'>","<class 'sklearn.kernel_ridge.KernelRidge'>","<class 'sklearn.linear_model.coordinate_descent.Lasso'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>","<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>","<class 'sklearn.linear_model.least_angle.Lars'>","<class 'sklearn.linear_model.least_angle.LassoLars'>","<class 'sklearn.linear_model.bayes.BayesianRidge'>","<class 'sklearn.linear_model.bayes.ARDRegression'>","<class 'sklearn.linear_model.logistic.LogisticRegression'>","<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>","<class 'sklearn.svm.classes.SVR'>","<class 'sklearn.svm.classes.NuSVR'>","<class 'sklearn.svm.classes.LinearSVR'>","<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",))
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics.scorer._PredictScorer'>",))
    class Outputs:
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class cross_val_score(object):
    task = 'Search'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_score'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", ("<class 'sklearn.linear_model.base.LinearRegression'>","<class 'sklearn.linear_model.ridge.Ridge'>","<class 'sklearn.kernel_ridge.KernelRidge'>","<class 'sklearn.linear_model.coordinate_descent.Lasso'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>","<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>","<class 'sklearn.linear_model.least_angle.Lars'>","<class 'sklearn.linear_model.least_angle.LassoLars'>","<class 'sklearn.linear_model.bayes.BayesianRidge'>","<class 'sklearn.linear_model.bayes.ARDRegression'>","<class 'sklearn.linear_model.logistic.LogisticRegression'>","<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>","<class 'sklearn.svm.classes.SVR'>","<class 'sklearn.svm.classes.NuSVR'>","<class 'sklearn.svm.classes.LinearSVR'>","<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",))
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics.scorer._PredictScorer'>",))
    class Outputs:
        scores = Output("scores","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class learning_curve(object):
    task = 'Search'
    subtask = 'grid'
    host = 'sklearn'
    function = 'learning_curve'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", ("<class 'sklearn.linear_model.base.LinearRegression'>","<class 'sklearn.linear_model.ridge.Ridge'>","<class 'sklearn.kernel_ridge.KernelRidge'>","<class 'sklearn.linear_model.coordinate_descent.Lasso'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>","<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>","<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>","<class 'sklearn.linear_model.least_angle.Lars'>","<class 'sklearn.linear_model.least_angle.LassoLars'>","<class 'sklearn.linear_model.bayes.BayesianRidge'>","<class 'sklearn.linear_model.bayes.ARDRegression'>","<class 'sklearn.linear_model.logistic.LogisticRegression'>","<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>","<class 'sklearn.svm.classes.SVR'>","<class 'sklearn.svm.classes.NuSVR'>","<class 'sklearn.svm.classes.LinearSVR'>","<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",))
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics.scorer._PredictScorer'>",))
        cv = Input("cv","instance of scikit-learn's cross validation generator", ("<class 'sklearn.model_selection._split.KFold'>","<class 'sklearn.model_selection._split.ShuffleSplit'>","<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>",))
    class Outputs:
        train_sizes_abs = Output("train_sizes_abs","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        extended_result_ = Output("extended_result_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        test_scores = Output("test_scores","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        train_scores = Output("train_scores","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class evaluate_regression(object):
    task = 'Search'
    subtask = 'evaluate'
    host = 'sklearn'
    function = 'evaluate_regression'
    modules = ('sklearn','metrics')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy_predict = Input("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        evaluation_results_ = Output("evaluation_results_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        evaluator = Output("evaluator","dictionary of metrics and their score function", ("<type 'dict'>",))
    class WParameters:
        mae_multioutput = Parameter('mae_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = ('raw_values', 'uniform_average'))
        r2_score = Parameter('r2_score','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                        description = "",
                        options = (True, False))
        mean_absolute_error = Parameter('mean_absolute_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = (True, False))
        multioutput = Parameter('multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                        description = "",
                        options = ('raw_values', 'uniform_average', 'variance_weighted'))
        mse_sample_weight = Parameter('mse_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = [])
        rmse_multioutput = Parameter('rmse_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = ('raw_values', 'uniform_average'))
        median_absolute_error = Parameter('median_absolute_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error',
                        description = "",
                        options = (True, False))
        mae_sample_weight = Parameter('mae_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = [])
        rmse_sample_weight = Parameter('rmse_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = [])
        mean_squared_error = Parameter('mean_squared_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = (True, False))
        root_mean_squared_error = Parameter('root_mean_squared_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = (True, False))
        explained_variance_score = Parameter('explained_variance_score','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score',
                        description = "",
                        options = (True, False))
        r2_sample_weight = Parameter('r2_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                        description = "",
                        options = [])
        ev_sample_weight = Parameter('ev_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score',
                        description = "",
                        options = [])
        ev_multioutput = Parameter('ev_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score',
                        description = "",
                        options = ('raw_values', 'uniform_average', 'variance_weighted'))
        mse_multioutput = Parameter('mse_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = ('raw_values', 'uniform_average'))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass
class scorer_regression(object):
    task = 'Search'
    subtask = 'evaluate'
    host = 'sklearn'
    function = 'scorer_regression'
    modules = ('sklearn','metrics')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/0.15/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer"

    class Inputs:
        pass

    class Outputs:
        scorer = Output("scorer","Callable object that returns a scalar score", ("<class 'sklearn.metrics.scorer._PredictScorer'>",))

    class WParameters:
        metric = Parameter('metric','mae','http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics',
                        description = "",
                        options = ('mae', 'mse', 'rmse', 'r2'))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        pass


