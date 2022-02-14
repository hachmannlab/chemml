from chemml.wrapper.database.containers import Input, Output, Parameter, req, regression_types, cv_classes
# import warnings
# warnings.filterwarnings("ignore")

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
        threshold = Parameter('threshold', 0.0)
        copy = Parameter('copy', True)


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
        include_bias = Parameter('include_bias', True)
        interaction_only = Parameter('interaction_only', False)
        degree = Parameter('degree', 2)
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
        dtype = Parameter('dtype', "<type'numpy.float64'>")
        n_values = Parameter('n_values', 'auto')
        sparse = Parameter('sparse', True)
        categorical_features = Parameter('categorical_features', 'all')
        handle_unknown = Parameter('handle_unknown', 'error')
class Imputer(object):
    task = 'Prepare'
    subtask = 'data cleaning'
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
        missing_values = Parameter('missing_values', "NaN")
        copy = Parameter('copy', True)
        verbose = Parameter('verbose', 0)
        strategy = Parameter('strategy', 'mean')
        axis = Parameter('axis', 0)

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
        fit_inverse_transform = Parameter('fit_inverse_transform', False)
        kernel = Parameter('kernel', 'linear')
        n_jobs = Parameter('n_jobs', 1)
        remove_zero_eig = Parameter('remove_zero_eig', False)
        degree = Parameter('degree', 3)
        max_iter = Parameter('max_iter', None)
        kernel_params = Parameter('kernel_params', None)
        random_state = Parameter('random_state', None)
        n_components = Parameter('n_components', None)
        eigen_solver = Parameter('eigen_solver', 'auto')
        tol = Parameter('tol', 0)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
        coef0 = Parameter('coef0', 1)
        gamma = Parameter('gamma', None)
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
        iterated_power = Parameter('iterated_power', 'auto')
        random_state = Parameter('random_state', None)
        whiten = Parameter('whiten', False)
        n_components = Parameter('n_components', None)
        tol = Parameter('tol', 0.0)
        copy = Parameter('copy', True)
        svd_solver = Parameter('svd_solver', 'auto')

class Normalizer(object):
    task = 'Prepare'
    subtask = 'scaling'
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
        copy = Parameter('copy', True)
        norm = Parameter('norm', 'l2')
class StandardScaler(object):
    task = 'Prepare'
    subtask = 'scaling'
    host = 'sklearn'
    function = 'StandardScaler'
    modules = ('sklearn','preprocessing')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing._data.StandardScaler'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's StandardScaler class", ("<class 'sklearn.preprocessing._data.StandardScaler'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api ",
                        options = ('fit_transform', 'transform', 'inverse_transform', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        copy = Parameter('copy', True)
        with_mean = Parameter('with_mean', True)
        with_std = Parameter('with_std', True)
class MinMaxScaler(object):
    task = 'Prepare'
    subtask = 'scaling'
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
        copy = Parameter('copy', True)
        feature_range = Parameter('feature_range', (0, 1))
class MaxAbsScaler(object):
    task = 'Prepare'
    subtask = 'scaling'
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
        copy = Parameter('copy', True)
class RobustScaler(object):
    task = 'Prepare'
    subtask = 'scaling'
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
        with_centering = Parameter('with_centering', True)
        copy = Parameter('copy', True)
        with_scaling = Parameter('with_scaling', True)
        quantile_range = Parameter('quantile_range', (25.0, 75.0))

class ShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'ShuffleSplit'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit"

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
        api = Output("api","instance of scikit-learn's ShuffleSplit class", ("<class 'sklearn.model_selection._split.ShuffleSplit'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        n_splits = Parameter('n_splits', 10)
        test_size = Parameter('test_size', 'default')
        train_size = Parameter('train_size', None)
        random_state = Parameter('random_state', None)

class StratifiedShuffleSplit(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'StratifiedShuffleSplit'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit"

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
        api = Output("api","instance of scikit-learn's StratifiedShuffleSplit class", ("<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        n_splits = Parameter('n_splits', 10)
        test_size = Parameter('test_size', 'default')
        train_size = Parameter('train_size', None)
        random_state = Parameter('random_state', None)
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
        test_size = Parameter('test_size', 0.25)
        train_size = Parameter('train_size', None)
        random_state = Parameter('random_state', None)
        shuffle = Parameter('shuffle', True)
        stratify = Parameter('stratify', None)
class KFold(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'KFold'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html"

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's KFold class", ("<class 'sklearn.model_selection._split.KFold'>",))
        fold_gen = Output("fold_gen","Generator of indices to split data into training and test set", ("<type 'generator'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('split', None))
    class FParameters:
        n_splits = Parameter('n_splits', 3)
        random_state = Parameter('random_state', None)
        shuffle = Parameter('shuffle', False)
class LeaveOneOut(object):
    task = 'Prepare'
    subtask = 'split'
    host = 'sklearn'
    function = 'LeaveOneOut'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html"

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's LeaveOneOut class", ("<class 'sklearn.model_selection._split.LeaveOneOut'>",))
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
        normalize = Parameter('normalize', False)
        n_iter = Parameter('n_iter', 300)
        verbose = Parameter('verbose', False)
        lambda_1 = Parameter('lambda_1', 1e-06)
        lambda_2 = Parameter('lambda_2', 1e-06)
        fit_intercept = Parameter('fit_intercept', True)
        threshold_lambda = Parameter('threshold_lambda', 10000.0)
        alpha_2 = Parameter('alpha_2', 1e-06)
        tol = Parameter('tol', 0.001)
        alpha_1 = Parameter('alpha_1', 1e-06)
        copy_X = Parameter('copy_X', True)
        compute_score = Parameter('compute_score', False)
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
        normalize = Parameter('normalize', False)
        n_iter = Parameter('n_iter', 300)
        verbose = Parameter('verbose', False)
        lambda_1 = Parameter('lambda_1', 1e-06)
        lambda_2 = Parameter('lambda_2', 1e-06)
        fit_intercept = Parameter('fit_intercept', True)
        alpha_2 = Parameter('alpha_2', 1e-06)
        tol = Parameter('tol', 0.001)
        alpha_1 = Parameter('alpha_1', 1e-06)
        copy_X = Parameter('copy_X', True)
        compute_score = Parameter('compute_score', False)
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
        normalize = Parameter('normalize', False)
        warm_start = Parameter('warm_start', False)
        selection = Parameter('selection', 'cyclic')
        fit_intercept = Parameter('fit_intercept', True)
        l1_ratio = Parameter('l1_ratio', 0.5)
        max_iter = Parameter('max_iter', 1000)
        precompute = Parameter('precompute', False)
        random_state = Parameter('random_state', None)
        tol = Parameter('tol', 0.0001)
        positive = Parameter('positive', False)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
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
        kernel = Parameter('kernel', 'linear')
        degree = Parameter('degree', 3)
        kernel_params = Parameter('kernel_params', None)
        alpha = Parameter('alpha', 1)
        coef0 = Parameter('coef0', 1)
        gamma = Parameter('gamma', None)
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
        n_nonzero_coefs = Parameter('n_nonzero_coefs', 500)
        normalize = Parameter('normalize', True)
        fit_path = Parameter('fit_path', True)
        fit_intercept = Parameter('fit_intercept', True)
        positive = Parameter('positive', False)
        eps = Parameter('eps', 2.22044604925e-16)
        precompute = Parameter('precompute', 'auto')
        copy_X = Parameter('copy_X', True)
        verbose = Parameter('verbose', False)
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
        normalize = Parameter('normalize', False)
        warm_start = Parameter('warm_start', False)
        selection = Parameter('selection', 'cyclic')
        fit_intercept = Parameter('fit_intercept', True)
        positive = Parameter('positive', False)
        max_iter = Parameter('max_iter', 1000)
        precompute = Parameter('precompute', False)
        random_state = Parameter('random_state', None)
        tol = Parameter('tol', 0.0001)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
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
        normalize = Parameter('normalize', True)
        fit_path = Parameter('fit_path', True)
        fit_intercept = Parameter('fit_intercept', True)
        positive = Parameter('positive', False)
        max_iter = Parameter('max_iter', 500)
        eps = Parameter('eps', 2.22044604925e-16)
        precompute = Parameter('precompute', 'auto')
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
        verbose = Parameter('verbose', False)
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
        copy_X = Parameter('copy_X', True)
        normalize = Parameter('normalize', False)
        n_jobs = Parameter('n_jobs', 1)
        fit_intercept = Parameter('fit_intercept', True)
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
        loss = Parameter('loss', 'epsilon_insensitive')
        C = Parameter('C', 1.0)
        intercept_scaling = Parameter('intercept_scaling', 1.0)
        fit_intercept = Parameter('fit_intercept', True)
        epsilon = Parameter('epsilon', 0.0)
        max_iter = Parameter('max_iter', 1000)
        random_state = Parameter('random_state', None)
        dual = Parameter('dual', True)
        tol = Parameter('tol', 0.0001)
        verbose = Parameter('verbose', 0)
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
        warm_start = Parameter('warm_start', False)
        C = Parameter('C', 1.0)
        n_jobs = Parameter('n_jobs', 1)
        verbose = Parameter('verbose', 0)
        intercept_scaling = Parameter('intercept_scaling', 1)
        fit_intercept = Parameter('fit_intercept', True)
        max_iter = Parameter('max_iter', 100)
        penalty = Parameter('penalty', 'l2')
        multi_class = Parameter('multi_class', 'ovr')
        random_state = Parameter('random_state', None)
        dual = Parameter('dual', False)
        tol = Parameter('tol', 0.0001)
        solver = Parameter('solver', 'liblinear')
        class_weight = Parameter('class_weight', None)
class MLPRegressor(object):
    task = 'Model'
    subtask = 'regression'
    host = 'sklearn'
    function = 'MLPRegressor'
    modules = ('sklearn','neural_network')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor"

    class Inputs:
        api = Input("api","instance of scikit-learn's MLPRegressor class", ("<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>",))
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        api = Output("api","instance of scikit-learn's MLPRegressor class", ("<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>",))
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        beta_1 = Parameter('beta_1', 0.9)
        warm_start = Parameter('warm_start', False)
        beta_2 = Parameter('beta_2', 0.999)
        shuffle = Parameter('shuffle', True)
        verbose = Parameter('verbose', False)
        nesterovs_momentum = Parameter('nesterovs_momentum', True)
        hidden_layer_sizes = Parameter('hidden_layer_sizes', (100,))
        epsilon = Parameter('epsilon', 1e-08)
        activation = Parameter('activation', 'relu')
        max_iter = Parameter('max_iter', 200)
        batch_size = Parameter('batch_size', 'auto')
        power_t = Parameter('power_t', 0.5)
        random_state = Parameter('random_state', None)
        learning_rate_init = Parameter('learning_rate_init', 0.001)
        tol = Parameter('tol', 0.0001)
        validation_fraction = Parameter('validation_fraction', 0.1)
        alpha = Parameter('alpha', 0.0001)
        solver = Parameter('solver', 'adam')
        momentum = Parameter('momentum', 0.9)
        learning_rate = Parameter('learning_rate', 'constant')
        early_stopping = Parameter('early_stopping', False)
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
        normalize = Parameter('normalize', False)
        warm_start = Parameter('warm_start', False)
        selection = Parameter('selection', 'cyclic')
        fit_intercept = Parameter('fit_intercept', True)
        l1_ratio = Parameter('l1_ratio', 0.5)
        max_iter = Parameter('max_iter', 1000)
        random_state = Parameter('random_state', None)
        tol = Parameter('tol', 0.0001)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
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
        normalize = Parameter('normalize', False)
        warm_start = Parameter('warm_start', False)
        selection = Parameter('selection', 'cyclic')
        fit_intercept = Parameter('fit_intercept', True)
        max_iter = Parameter('max_iter', 1000)
        random_state = Parameter('random_state', None)
        tol = Parameter('tol', 0.0001)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
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
        kernel = Parameter('kernel', 'rbf')
        C = Parameter('C', 1.0)
        verbose = Parameter('verbose', False)
        degree = Parameter('degree', 3)
        shrinking = Parameter('shrinking', True)
        max_iter = Parameter('max_iter', -1)
        tol = Parameter('tol', 0.001)
        cache_size = Parameter('cache_size', 200)
        coef0 = Parameter('coef0', 0.0)
        nu = Parameter('nu', 0.5)
        gamma = Parameter('gamma', 'auto')
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
        normalize = Parameter('normalize', False)
        fit_intercept = Parameter('fit_intercept', True)
        max_iter = Parameter('max_iter', None)
        random_state = Parameter('random_state', None)
        tol = Parameter('tol', 0.001)
        copy_X = Parameter('copy_X', True)
        alpha = Parameter('alpha', 1.0)
        solver = Parameter('solver', 'auto')
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
        warm_start = Parameter('warm_start', False)
        loss = Parameter('loss', 'squared_loss')
        eta0 = Parameter('eta0', 0.01)
        verbose = Parameter('verbose', 0)
        shuffle = Parameter('shuffle', True)
        fit_intercept = Parameter('fit_intercept', True)
        l1_ratio = Parameter('l1_ratio', 0.15)
        average = Parameter('average', False)
        n_iter = Parameter('n_iter', 5)
        penalty = Parameter('penalty', 'l2')
        power_t = Parameter('power_t', 0.25)
        random_state = Parameter('random_state', None)
        epsilon = Parameter('epsilon', 0.1)
        alpha = Parameter('alpha', 0.0001)
        learning_rate = Parameter('learning_rate', 'invscaling')
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
        kernel = Parameter('kernel', 'rbf')
        C = Parameter('C', 1.0)
        verbose = Parameter('verbose', False)
        degree = Parameter('degree', 3)
        epsilon = Parameter('epsilon', 0.1)
        shrinking = Parameter('shrinking', True)
        max_iter = Parameter('max_iter', -1)
        tol = Parameter('tol', 0.001)
        cache_size = Parameter('cache_size', 200)
        coef0 = Parameter('coef0', 0.0)
        gamma = Parameter('gamma', 'auto')

class GridSearchCV(object):
    task = 'Optimize'
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
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics._scorer._PredictScorer'>",))
        cv = Input("cv", "instance of scikit-learn's cross validation generator or instance object", tuple(["<type 'generator'>"]+list(cv_classes())))

    class Outputs:
        cv_results_ = Output("cv_results_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of scikit-learn's GridSearchCV class", ("<class 'sklearn.grid_search.GridSearchCV'>",))
        best_estimator_ = Output("best_estimator_","instance of a machine learning class", regression_types())
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        scoring = Parameter('scoring', None)
        n_jobs = Parameter('n_jobs', 1)
        verbose = Parameter('verbose', 0)
        fit_params = Parameter('fit_params', None)
        refit = Parameter('refit', True)
        return_train_score = Parameter('return_train_score', True)
        iid = Parameter('iid', True)
        estimator = Parameter('estimator', '@estimator', required=True)
        error_score = Parameter('error_score','raise')
        pre_dispatch = Parameter('pre_dispatch', '2 * n_jobs')
        param_grid = Parameter('param_grid', {}, required=True)
        cv = Parameter('cv', None)

class cross_val_predict(object):
    task = 'Optimize'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_predict'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", regression_types())
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics._scorer._PredictScorer'>",))
        cv = Input("cv","cross-validation generator or instance object", tuple(["<type 'generator'>"]+list(cv_classes())))
    class Outputs:
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        estimator = Parameter('estimator', '@estimator', required=True)
        X = Parameter('X', '@dfx', required=True)
        y = Parameter('y', None)
        groups = Parameter('groups', None)
        cv = Parameter('cv', None)
        n_jobs = Parameter('n_jobs', 1)
        verbose = Parameter('verbose', 0)
        fit_params = Parameter('fit_params', None)
        pre_dispatch = Parameter('pre_dispatch', '2 * n_jobs')
        method = Parameter('method', 'predict')
class cross_val_score(object):
    task = 'Optimize'
    subtask = 'validate'
    host = 'sklearn'
    function = 'cross_val_score'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", regression_types())
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics._scorer._PredictScorer'>",))
        cv = Input("cv", "cross-validation generator or instance object", tuple(["<type 'generator'>"]+list(cv_classes())))

    class Outputs:
        scores = Output("scores","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        estimator = Parameter('estimator', '@estimator', required=True)
        X = Parameter('X', '@dfx', required=True)
        y = Parameter('y', None)
        groups = Parameter('groups', None)
        scoring = Parameter('scoring', None)
        cv = Parameter('cv', None)
        n_jobs = Parameter('n_jobs', 1)
        verbose = Parameter('verbose', 0)
        fit_params = Parameter('fit_params', None)
        pre_dispatch = Parameter('pre_dispatch', '2 * n_jobs')
class learning_curve(object):
    task = 'Optimize'
    subtask = 'grid'
    host = 'sklearn'
    function = 'learning_curve'
    modules = ('sklearn','model_selection')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve"

    class Inputs:
        dfy = Input("dfy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        estimator = Input("estimator","instance of a machine learning class", regression_types())
        scorer = Input("scorer","instance of scikit-learn's make_scorer class", ("<class 'sklearn.metrics._scorer._PredictScorer'>",))
        cv = Input("cv","instance of scikit-learn's cross validation generator or instance object",
                   tuple(["<type 'generator'>"] + list(cv_classes())))
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
        estimator = Parameter('estimator', '@estimator', required=True)
        X = Parameter('X', '@dfx', required=True)
        y = Parameter('y', None)
        groups = Parameter('groups', None)
        train_sizes = Parameter('train_sizes', [ 0.1, 0.33, 0.55, 0.78,1.])
        scoring = Parameter('scoring', None)
        cv = Parameter('cv', None)
        exploit_incremental_learning = Parameter('exploit_incremental_learning', False)
        n_jobs = Parameter('n_jobs', 1)
        pre_dispatch = Parameter('pre_dispatch', 'all')
        verbose = Parameter('verbose', 0)
        shuffle = Parameter('shuffle', False)
        random_state = Parameter('random_state', None)
class evaluate_regression(object):
    task = 'Optimize'
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
        r2_score = Parameter('r2_score','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                        description = "",
                        options = (True, False))
        r2_sample_weight = Parameter('r2_sample_weight', 'None',
                                     'http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                                     description="",
                                     options=[])
        multioutput = Parameter('multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score',
                        description = "",
                        options = ('raw_values', 'uniform_average', 'variance_weighted'))

        mean_absolute_error = Parameter('mean_absolute_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = (True, False))
        mae_multioutput = Parameter('mae_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = ('raw_values', 'uniform_average'))
        mae_sample_weight = Parameter('mae_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error',
                        description = "",
                        options = [])

        mse_sample_weight = Parameter('mse_sample_weight','None','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = [])
        rmse_multioutput = Parameter('rmse_multioutput','uniform_average','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error',
                        description = "",
                        options = ('raw_values', 'uniform_average'))
        median_absolute_error = Parameter('median_absolute_error','False','http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error',
                        description = "",
                        options = (True, False))
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
    task = 'Optimize'
    subtask = 'evaluate'
    host = 'sklearn'
    function = 'scorer_regression'
    modules = ('sklearn','metrics')
    requirements = (req(1), req(2))
    documentation = "http://scikit-learn.org/0.15/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer"

    class Inputs:
        pass

    class Outputs:
        scorer = Output("scorer","Callable object that returns a scalar score", ("<class 'sklearn.metrics._scorer._PredictScorer'>",))

    class WParameters:
        metric = Parameter('metric','mae',
                        description = "http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics",
                        format = "string: 'mae', 'mse', 'r2'",
                        options = ('mae', 'mse', 'r2'))
        track_header = Parameter('track_header', True,'Boolean',
                        description = "if True, the input dataframe's header will be transformed to the output dataframe",
                        options = (True, False))
    class FParameters:
        greater_is_better = Parameter('greater_is_better', True)
        needs_proba = Parameter('needs_proba', False)
        needs_threshold = Parameter('needs_threshold', False)
        kwargs = Parameter('kwargs', {}, format = 'dictionary')