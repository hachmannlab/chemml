.. _learning_curve:

learning_curve
===============

:task:
    | Search

:subtask:
    | grid

:host:
    | sklearn

:function:
    | learning_curve

:input tokens (receivers):
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``estimator`` : instance of a machine learning class
    |   ("<class 'sklearn.linear_model.base.LinearRegression'>", "<class 'sklearn.linear_model.ridge.Ridge'>", "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>", "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>", "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>", "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>", "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>")
    | ``scorer`` : instance of scikit-learn's make_scorer class
    |   ("<class 'sklearn.metrics.scorer._PredictScorer'>",)
    | ``cv`` : instance of scikit-learn's cross validation generator
    |   ("<class 'sklearn.model_selection._split.KFold'>", "<class 'sklearn.model_selection._split.ShuffleSplit'>", "<class 'sklearn.model_selection._split.StratifiedShuffleSplit'>")

:output tokens (senders):
    | ``train_sizes_abs`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``extended_result_`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``test_scores`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``train_scores`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = learning_curve``
    |   ``<< track_header = True``
    |   ``<< scoring = None``
    |   ``<< n_jobs = 1``
    |   ``<< shuffle = False``
    |   ``<< groups = None``
    |   ``<< random_state = None``
    |   ``<< pre_dispatch = all``
    |   ``<< estimator = required_required``
    |   ``<< exploit_incremental_learning = False``
    |   ``<< train_sizes = [0.1, 0.33, 0.55, 0.78, 1.0]``
    |   ``<< y = None``
    |   ``<< X = required_required``
    |   ``<< cv = None``
    |   ``<< verbose = 0``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id estimator``
    |   ``>> id scorer``
    |   ``>> id cv``
    |   ``>> id train_sizes_abs``
    |   ``>> id extended_result_``
    |   ``>> id test_scores``
    |   ``>> id train_scores``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve