.. _GridSearchCV:

GridSearchCV
=============

:task:
    | Search

:subtask:
    | grid

:host:
    | sklearn

:function:
    | GridSearchCV

:input tokens (receivers):
    | ``dfy`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``estimator`` : instance of a machine learning class
    |   types: ("<type 'str'>", "<class 'sklearn.linear_model.base.LinearRegression'>", "<class 'sklearn.linear_model.ridge.Ridge'>", "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>", "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>", "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>", "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>", "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>", "<class 'cheml.nn.keras.mlp.MLP_sklearn'>")
    | ``scorer`` : instance of scikit-learn's make_scorer class
    |   types: ("<class 'sklearn.metrics.scorer._PredictScorer'>",)
    | ``cv`` : instance of scikit-learn's cross validation generator or instance object
    |   types: ("<type 'generator'>", "<class 'sklearn.model_selection._split.KFold'>", "<class 'sklearn.model_selection._split.ShuffleSplit'>", "<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>", "<class 'sklearn.model_selection._split.LeaveOneOut'>")

:output tokens (senders):
    | ``cv_results_`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's GridSearchCV class
    |   types: ("<class 'sklearn.grid_search.GridSearchCV'>",)
    | ``best_estimator_`` : instance of a machine learning class
    |   types: ("<class 'sklearn.linear_model.base.LinearRegression'>", "<class 'sklearn.linear_model.ridge.Ridge'>", "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>", "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>", "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>", "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>", "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>", "<class 'cheml.nn.keras.mlp.MLP_sklearn'>")

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = GridSearchCV``
    |   ``<< track_header = True``
    |   ``<< scoring = None``
    |   ``<< n_jobs = 1``
    |   ``<< verbose = 0``
    |   ``<< fit_params = None``
    |   ``<< refit = True``
    |   ``<< return_train_score = True``
    |   ``<< iid = True``
    |   ``<< estimator = @estimator``
    |   ``<< error_score = raise``
    |   ``<< pre_dispatch = 2 * n_jobs``
    |   ``<< param_grid = {}``
    |   ``<< cv = None``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id estimator``
    |   ``>> id scorer``
    |   ``>> id cv``
    |   ``>> id cv_results_``
    |   ``>> id api``
    |   ``>> id best_estimator_``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
