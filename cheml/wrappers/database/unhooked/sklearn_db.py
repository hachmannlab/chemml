import numpy as np
from cheml.wrappers.database.containers import Input, Output, Parameter, req, regression_types, cv_types

class mask(object):
    task = ""
    subtask = ""
    host = ""
    function = ""
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's X class",())

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's X class",())

    class WParameters:
        """ Wrapper parameters"""
        func_method = Parameter('func_method', None, 'string')

    class Fparameters:
        """ Function parameters"""
        #
        pass

class PolynomialFeatures(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'PolynomialFeatures'
    module = ('sklearn', 'preprocessing')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's PolynomialFeatures class",
                    ["<class 'sklearn.preprocessing.data.PolynomialFeatures'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's PolynomialFeatures class",
                    ["<class 'sklearn.preprocessing.data.PolynomialFeatures'>"])

    class WParameters:
        """ Wrapper parameters"""
        func_method = Parameter('func_method', None, 'string',
                                description = "fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                                options=('fit_transform', 'transform', None))

    class Fparameters:
        """ Function parameters"""
        #{'include_bias': True, 'interaction_only': False, 'degree': 2}
        pass

class Binarizer(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'Binarizer'
    module = ('sklearn', 'preprocessing')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Binarizer class",
                    ["<class 'sklearn.preprocessing.data.Binarizer'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Binarizer class",
                    ["<class 'sklearn.preprocessing.data.Binarizer'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                                options=('fit_transform', 'transform', None))
    class Fparameters:
        #{'threshold': 0.0, 'copy': True}
        pass

class OneHotEncoder(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'sklearn'
    function = 'OneHotEncoder'
    module = ('sklearn', 'preprocessing')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's OneHotEncoder class",
                    ["<class 'sklearn.preprocessing.data.OneHotEncoder'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's OneHotEncoder class",
                    ["<class 'sklearn.preprocessing.data.OneHotEncoder'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                                options=('fit_transform', 'transform', None))
    class Fparameters:
        # {'dtype': np.float, 'handle_unknown': 'error', 'sparse': True, 'categorical_features': 'all', 'n_values': 'auto'}
        pass

class Imputer(object):
    task = 'Prepare'
    subtask = 'preprocessor'
    host = 'sklearn'
    function = 'Imputer'
    module = ('sklearn', 'preprocessing')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Imputer class",
                    ["<class 'sklearn.preprocessing.imputation.Imputer'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Imputer class",
                    ["<class 'sklearn.preprocessing.imputation.Imputer'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; None: only make a new api ",
                                options=('fit_transform', 'transform', None))

    class Fparameters:
        # {'copy': True, 'strategy': 'mean', 'axis': 0, 'verbose': 0, 'missing_values': 'NaN'}
        pass

class StandardScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'StandardScaler'
    module = ('sklearn', 'preprocessing')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's StandardScaler class",
                    ["<class 'sklearn.preprocessing.data.StandardScaler'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's StandardScaler class",
                    ["<class 'sklearn.preprocessing.data.StandardScaler'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))

    class Fparameters:
        # {'copy': True, 'with_mean': True, 'with_std': True}
        pass

class MinMaxScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'MinMaxScaler'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MinMaxScaler class",
                    ["<class 'sklearn.preprocessing.data.MinMaxScaler'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MinMaxScaler class",
                    ["<class 'sklearn.preprocessing.data.MinMaxScaler'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))

    class Fparameters:
        # {'copy': True, 'feature_range': (0, 1)}
        pass

class MaxAbsScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'MaxAbsScaler'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MaxAbsScaler class",
                    ["<class 'sklearn.preprocessing.data.MaxAbsScaler'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MaxAbsScaler class",
                    ["<class 'sklearn.preprocessing.data.MaxAbsScaler'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))

    class Fparameters:
        # {'copy':True}
        pass

class RobustScaler(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'RobustScaler'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's RobustScaler class",
                    ["<class 'sklearn.preprocessing.data.RobustScaler'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's RobustScaler class",
                    ["<class 'sklearn.preprocessing.data.RobustScaler'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))

    class Fparameters:
        # {'with_centering': True, 'quantile_range': (25.0, 75.0), 'copy': True, 'with_scaling': True}
        pass

class Normalizer(object):
    task = 'Prepare'
    subtask = 'scale'
    host = 'sklearn'
    function = 'Normalizer'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Normalizer class",
                    ["<class 'sklearn.preprocessing.data.Normalizer'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Normalizer class",
                    ["<class 'sklearn.preprocessing.data.Normalizer'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api None: only make a new api ",
                                options = ('fit_transform','transform',None))

    class Fparameters:
        # {'copy': True, 'norm': 'l2'}
        pass

class PCA(object):
    task = 'Prepare'
    subtask = 'feature transformation'
    host = 'sklearn'
    function = 'PCA'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's PCA class",
                    ["<class 'sklearn.decomposition.pca.PCA'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's PCA class",
                    ["<class 'sklearn.decomposition.pca.PCA'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))


    class Fparameters:
        # {'copy': True, 'norm': 'l2'}
        pass

class KernelPCA(object):
    task = 'Prepare'
    subtask = 'feature transformation'
    host = 'sklearn'
    function = 'KernelPCA'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA"

    class Inputs:
        df = Input('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's KernelPCA class",
                    ["<class 'sklearn.decomposition.kernel_pca.KernelPCA'>"])

    class Outputs:
        df = Output('df', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's KernelPCA class",
                    ["<class 'sklearn.decomposition.kernel_pca.KernelPCA'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                description="fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                                options = ('fit_transform','transform','inverse_transform',None))

    class Fparameters:
        # {'fit_inverse_transform': False, 'kernel': 'linear', 'n_jobs': 1, 'tol': 0, 'degree': 3, 'random_state': None, 'max_iter': None, 'kernel_params': None, 'remove_zero_eig': False, 'n_components': None, 'eigen_solver': 'auto', 'copy_X': True, 'alpha': 1.0, 'coef0': 1, 'gamma': None}
        pass

class train_test_split(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'train_test_split'
    module = ('sklearn','model_selection')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class Outputs:
        dfx_train = Output('dfx_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfx_test = Output('dfx_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_train = Output('dfy_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_test = Output('dfy_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class WParameters:
        pass

    class Fparameters:
        #
        pass

class KFold(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'KFold'
    module = ('sklearn','model_selection')
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html"

    class Inputs:
        df = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class Outputs:
        fold_gen = Output('cv_gen','Generator of indices to split data into training and test set',
                        ["<type 'generator'>"])
        api = Output('api', "instance of scikit-learn's KFold class",
                    ["<class 'sklearn.model_selection._split.KFold'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('split',None))

    class Fparameters:
        #n_splits=3, shuffle=False, random_state=None
        pass

class ShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'ShuffleSplit'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's ShuffleSplit class",
                    ["<class 'sklearn.model_selection._split.ShuffleSplit'>"])

    class Outputs:
        dfx_train = Output('dfx_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfx_test = Output('dfx_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_train = Output('dfy_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_test = Output('dfy_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's ShuffleSplit class",
                    ["<class 'sklearn.model_selection._split.ShuffleSplit'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('split',None))

    class Fparameters:
        # n_splits=10, test_size='default', train_size=None, random_state=None
        pass

class StratifiedShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'StratifiedShuffleSplit'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's StratifiedShuffleSplit class",
                    ["<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>"])

    class Outputs:
        dfx_train = Output('dfx_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfx_test = Output('dfx_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_train = Output('dfy_train', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy_test = Output('dfy_test', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's StratifiedShuffleSplit class",
                    ["<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('split',None))

    class Fparameters:
        # n_splits=10, test_size='default', train_size=None, random_state=None
        pass

class LinearRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LinearRegression'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's LinearRegression class",
                    ["<class 'sklearn.linear_model.base.LinearRegression'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's LinearRegression class",
                    ["<class 'sklearn.linear_model.base.LinearRegression'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        # {'fit_intercept': True, 'normalize': False, 'n_jobs': 1, 'copy_X': True}
        pass

class Ridge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Ridge'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Ridge class",
                    ["<class 'sklearn.linear_model.ridge.Ridge'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's Ridge class",
                    ["<class 'sklearn.linear_model.ridge.Ridge'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        # {'max_iter': None, 'normalize': False, 'tol': 0.001, 'alpha': 1.0, 'fit_intercept': True, 'copy_X': True, 'solver': 'auto', 'random_state': None}
        pass

class KernelRidge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'KernelRidge'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's KernelRidge class",
                    ["<class 'sklearn.kernel_ridge.KernelRidge'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's KernelRidge class",
                    ["<class 'sklearn.kernel_ridge.KernelRidge'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'gamma': None, 'coef0': 1, 'kernel_params': None, 'alpha': 1, 'degree': 3, 'kernel': 'linear'}
        pass

class Lasso(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Lasso'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Lasso class",
                    ["<class 'sklearn.linear_model.coordinate_descent.Lasso'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's Lasso class",
                    ["<class 'sklearn.linear_model.coordinate_descent.Lasso'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'positive': False, 'max_iter': 1000, 'normalize': False, 'tol': 0.0001, 'selection': 'cyclic', 'alpha': 1.0,
        # 'fit_intercept': True, 'copy_X': True, 'precompute': False, 'warm_start': False, 'random_state': None}
        pass

class MultiTaskLasso(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MultiTaskLasso'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MultiTaskLasso class",
                    ["<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's MultiTaskLasso class",
                    ["<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'max_iter': 1000, 'normalize': False, 'tol': 0.0001, 'selection': 'cyclic', 'alpha': 1.0, 'fit_intercept': True,
        #  'copy_X': True, 'warm_start': False, 'random_state': None}
        pass

class ElasticNet(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'ElasticNet'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's ElasticNet class",
                    ["<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's ElasticNet class",
                    ["<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'copy_X': True, 'tol': 0.0001, 'selection': 'cyclic', 'alpha': 1.0, 'fit_intercept': True, 'normalize': False,
        #  'random_state': None, 'max_iter': 1000, 'l1_ratio': 0.5, 'positive': False, 'precompute': False, 'warm_start': False}
        pass

class MultiTaskElasticNet(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MultiTaskElasticNet'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MultiTaskElasticNet class",
                    ["<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's MultiTaskElasticNet class",
                    ["<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'max_iter': 1000, 'copy_X': True, 'selection': 'cyclic', 'l1_ratio': 0.5, 'alpha': 1.0, 'fit_intercept': True, 'normalize': False, 'warm_start': False, 'random_state': None, 'tol': 0.0001}
        pass

class Lars(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'Lars'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's Lars class",
                    ["<class 'sklearn.linear_model.least_angle.Lars'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's Lars class",
                    ["<class 'sklearn.linear_model.least_angle.Lars'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'eps': 2.2204460492503131e-16, 'verbose': False, 'copy_X': True, 'positive': False, 'fit_intercept': True, 'normalize': True, 'precompute': 'auto', 'n_nonzero_coefs': 500, 'fit_path': True}
        pass

class LassoLars(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LassoLars'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's LassoLars class",
                    ["<class 'sklearn.linear_model.least_angle.LassoLars'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's LassoLars class",
                    ["<class 'sklearn.linear_model.least_angle.LassoLars'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'eps': 2.2204460492503131e-16, 'max_iter': 500, 'copy_X': True, 'alpha': 1.0, 'fit_intercept': True,
        # 'normalize': True, 'precompute': 'auto', 'verbose': False, 'fit_path': True, 'positive': False}
        pass

class BayesianRidge(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'BayesianRidge'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's BayesianRidge class",
                    ["<class 'sklearn.linear_model.bayes.BayesianRidge'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's BayesianRidge class",
                    ["<class 'sklearn.linear_model.bayes.BayesianRidge'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'alpha_1': 1e-06, 'n_iter': 300, 'lambda_1': 1e-06, 'copy_X': True, 'tol': 0.001, 'normalize': False,
        # 'compute_score': False, 'fit_intercept': True, 'alpha_2': 1e-06, 'verbose': False, 'lambda_2': 1e-06}
        pass

class ARDRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'ARDRegression'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's ARDRegression class",
                    ["<class 'sklearn.linear_model.bayes.ARDRegression'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's ARDRegression class",
                    ["<class 'sklearn.linear_model.bayes.ARDRegression'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'alpha_2': 1e-06, 'copy_X': True, 'tol': 0.001, 'fit_intercept': True, 'verbose': False, 'alpha_1': 1e-06,
        # 'lambda_1': 1e-06, 'threshold_lambda': 10000.0, 'compute_score': False, 'n_iter': 300, 'normalize': False, 'lambda_2': 1e-06}
        pass

class LogisticRegression(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LogisticRegression'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's LogisticRegression class",
                    ["<class 'sklearn.linear_model.logistic.LogisticRegression'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's LogisticRegression class",
                    ["<class 'sklearn.linear_model.logistic.LogisticRegression'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'class_weight': None, 'tol': 0.0001, 'dual': False, 'fit_intercept': True, 'solver': 'liblinear', 'random_state': None, 'multi_class': 'ovr', 'C': 1.0, 'max_iter': 100, 'verbose': 0, 'penalty': 'l2', 'n_jobs': 1, 'warm_start': False, 'intercept_scaling': 1}
        pass

class SGDRegressor(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'SGDRegressor'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's SGDRegressor class",
                    ["<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's SGDRegressor class",
                    ["<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'shuffle': True, 'loss': 'squared_loss', 'learning_rate': 'invscaling', 'alpha': 0.0001, 'fit_intercept': True,
        #  'epsilon': 0.1, 'random_state': None, 'penalty': 'l2', 'eta0': 0.01, 'l1_ratio': 0.15, 'verbose': 0, 'n_iter': 5, 'average': False, 'warm_start': False, 'power_t': 0.25}
        pass

class SVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'SVR'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's SVR class",
                    ["<class 'sklearn.svm.classes.SVR'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's SVR class",
                    ["<class 'sklearn.svm.classes.SVR'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'gamma': 'auto', 'coef0': 0.0, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'degree': 3, 'verbose': False,
        #  'C': 1.0, 'epsilon': 0.1, 'shrinking': True, 'kernel': 'rbf'}
        pass

class NuSVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'NuSVR'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's NuSVR class",
                    ["<class 'sklearn.svm.classes.NuSVR'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's NuSVR class",
                    ["<class 'sklearn.svm.classes.NuSVR'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'gamma': 'auto', 'coef0': 0.0, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'degree': 3, 'verbose': False,
        #  'C': 1.0, 'epsilon': 0.1, 'shrinking': True, 'kernel': 'rbf'}
        pass

class LinearSVR(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'LinearSVR'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's LinearSVR class",
                    ["<class 'sklearn.svm.classes.LinearSVR'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's LinearSVR class",
                    ["<class 'sklearn.svm.classes.LinearSVR'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'fit_intercept': True, 'max_iter': 1000, 'loss': 'epsilon_insensitive', 'tol': 0.0001, 'dual': True, 'C': 1.0,
        #  'verbose': 0, 'epsilon': 0.0, 'intercept_scaling': 1.0, 'random_state': None}
        pass

class MLPRegressor(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MLPRegressor'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Input('api', "instance of scikit-learn's MLPRegressor class",
                    ["<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's MLPRegressor class",
                    ["<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>"])

    class WParameters:
        func_method = Parameter('func_method', None, 'string',
                                options = ('fit','predict',None))

    class Fparameters:
        #{'beta_1': 0.9, 'warm_start': False, 'beta_2': 0.999, 'shuffle': True, 'verbose': False, 'nesterovs_momentum': True,
        #  'hidden_layer_sizes': (100,), 'epsilon': 1e-08, 'activation': 'relu', 'max_iter': 200, 'batch_size': 'auto',
        # 'power_t': 0.5, 'random_state': None, 'learning_rate_init': 0.001, 'tol': 0.0001, 'validation_fraction': 0.1,
        #  'alpha': 0.0001, 'solver': 'adam', 'momentum': 0.9, 'learning_rate': 'constant', 'early_stopping': False}
        pass

class GridSearchCV(object):
    task = 'Search'
    subtask = 'grid'
    host = 'sklearn'
    function = 'GridSearchCV'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        estimator = Input('estimator', "instance of a machine learning class", regression_types())
        scorer = Input('scorer', "instance of scikit-learn's make_scorer class",
                    ["<class 'sklearn.metrics.scorer._PredictScorer'>"])

    class Outputs:
        cv_results_ = Output('cv_results_', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        api = Output('api', "instance of scikit-learn's GridSearchCV class",
                    ["<class 'sklearn.grid_search.GridSearchCV'>"])
        best_estimator_ = Output('best_estimator_', "instance of a machine learning class",regression_types())

    class WParameters:
        pass

    class Fparameters:
        #estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True
        pass

class cross_val_score(object):
    task = 'Search'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_score'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        estimator = Input('estimator', "instance of a machine learning class", regression_types())
        scorer = Input('scorer', "instance of scikit-learn's make_scorer class",
                    ["<class 'sklearn.metrics.scorer._PredictScorer'>"])

    class Outputs:
        scores = Output('scores', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class WParameters:
        pass

    class Fparameters:
        # estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs'
        pass

class cross_val_predict(object):
    task = 'Search'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_predict'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        estimator = Input('estimator', "instance of a machine learning class", regression_types())
        scorer = Input('scorer', "instance of scikit-learn's make_scorer class",
                    ["<class 'sklearn.metrics.scorer._PredictScorer'>"])

    class Outputs:
        dfy_predict = Output('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class WParameters:
        pass

    class Fparameters:
        # estimator, X, y=None, groups=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', method='predict'
        pass

class learning_curve(object):
    task = 'Search'
    subtask = 'grid'
    host = 'sklearn'
    function = 'learning_curve'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve"

    class Inputs:
        dfx = Input('dfx', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        estimator = Input('estimator', "instance of a machine learning class", regression_types())
        scorer = Input('scorer', "instance of scikit-learn's make_scorer class",
                    ["<class 'sklearn.metrics.scorer._PredictScorer'>"])
        cv = Input('cv', "instance of scikit-learn's cross validation generator", cv_types())

    class Outputs:
        train_sizes_abs = Output('train_sizes_abs', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        train_scores = Output('train_scores', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        test_scores = Output('test_scores', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        extended_result_ = Output('extended_result_', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class WParameters:
        pass

    class Fparameters:
        # estimator, X, y, groups=None, train_sizes=array([ 0.1, 0.33, 0.55, 0.78, 1. ]), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=1, pre_dispatch='all', verbose=0, shuffle=False, random_state=None
        pass

class scorer_regression(object):
    task = 'Search'
    subtask = 'evaluate'
    host = 'sklearn'
    function = 'scorer_regression'
    requirements = (req(0))
    documentation = "http://scikit-learn.org/0.15/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer"

    class Inputs:
        pass

    class Outputs:
        scorer = Output('scorer', 'Callable object that returns a scalar score',
                        ["<class 'sklearn.metrics.scorer._PredictScorer'>"])

    class WParameters:
        metric = Parameter('metric', 'mae', "http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics",
                                options = ('mae','mse','rmse','r2'))

    class Fparameters:
        #score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs
        pass

class evaluate_regression(object):
    task = 'Search'
    subtask = 'evaluate'
    host = 'sklearn'
    function = 'evaluate_regression'
    requirements = (req(0), req(2))
    documentation = "http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics"

    class Inputs:
        dfy_predict = Input('dfy_predict', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        dfy = Input('dfy', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])

    class Outputs:
        evaluation_results_ = Output('evaluation_results_', 'pandas dataframe', ["<class 'pandas.core.frame.DataFrame'>"])
        evaluator = Output('evaluator', "dictionary of metrics and their score function",["<type 'dict'>"])

    class WParameters:
        r2_score = Parameter('r2_score', False,
                           "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score",
                           options=(True, False))
        r2_sample_weight = Parameter('r2_sample_weight', None,
                         "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score")
        multioutput = Parameter('multioutput', 'uniform_average',
                         "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score",
                         options=('raw_values', 'uniform_average', 'variance_weighted'))

        mean_absolute_error = Parameter('mean_absolute_error', False,
                             "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error",
                             options=(True, False))
        mae_sample_weight = Parameter('mae_sample_weight', None,
                                     "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error")
        mae_multioutput = Parameter('mae_multioutput', 'uniform_average',
                                "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error",
                                options=('raw_values', 'uniform_average'))

        median_absolute_error = Parameter('median_absolute_error', False,
                                    "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error",
                                    options=(True, False))

        mean_squared_error = Parameter('mean_squared_error', False,
                                      "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error",
                                      options=(True, False))
        mse_sample_weight = Parameter('mse_sample_weight', None,
                                          "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error")
        mse_multioutput = Parameter('mse_multioutput', 'uniform_average',
                                          "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error",
                                        options=('raw_values', 'uniform_average'))

        root_mean_squared_error = Parameter('root_mean_squared_error', False,
                                   "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error",
                                   options=(True, False))
        rmse_sample_weight = Parameter('rmse_sample_weight', None,
                                  "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error")
        rmse_multioutput = Parameter('rmse_multioutput', 'uniform_average',
                                "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error",
                                options=('raw_values', 'uniform_average'))

        explained_variance_score = Parameter('explained_variance_score', False,
                                        "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score",
                                        options=(True, False))
        ev_sample_weight = Parameter('ev_sample_weight', None,
                                   "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score")
        ev_multioutput = Parameter('ev_multioutput', 'uniform_average',
                                 "http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score",
                                 options=('raw_values', 'uniform_average', 'variance_weighted'))

    class Fparameters:
        pass




# [klass[0] for klass in inspect.getmembers(sklearn_db)] =
#['ARDRegression', 'BayesianRidge', 'Binarizer', 'ElasticNet', 'GridSearchCV', 'Imputer', 'Input', 'KFold', 'KernelPCA', 'KernelRidge', 'Lars', 'Lasso', 'LassoLars', 'LinearRegression', 'LinearSVR', 'LogisticRegression', 'MLPRegressor', 'MaxAbsScaler', 'MinMaxScaler', 'MultiTaskElasticNet', 'MultiTaskLasso', 'Normalizer', 'NuSVR', 'OneHotEncoder', 'Output', 'PCA', 'Parameter', 'PolynomialFeatures', 'Ridge', 'RobustScaler', 'SGDRegressor', 'SVR', 'ShuffleSplit', 'StandardScaler', 'StratifiedShuffleSplit', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'cross_val_predict', 'cross_val_score', 'cv_types', 'evaluate_regression', 'learning_curve', 'mask', 'np', 'regression_types', 'req', 'scorer_regression', 'train_test_split']
# len = all + 5 '__' + 6 imported + 1 np + 1 mask