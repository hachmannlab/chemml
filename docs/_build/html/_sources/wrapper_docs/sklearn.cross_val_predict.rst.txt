.. _cross_val_predict:

cross_val_predict
==================

:task:
    | Search

:subtask:
    | validate

:host:
    | sklearn

:function:
    | cross_val_predict

:input tokens (receivers):
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``estimator`` : instance of a machine learning class
    |   ("<class 'sklearn.linear_model.base.LinearRegression'>", "<class 'sklearn.linear_model.ridge.Ridge'>", "<class 'sklearn.kernel_ridge.KernelRidge'>", "<class 'sklearn.linear_model.coordinate_descent.Lasso'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>", "<class 'sklearn.linear_model.coordinate_descent.ElasticNet'>", "<class 'sklearn.linear_model.coordinate_descent.MultiTaskElasticNet'>", "<class 'sklearn.linear_model.least_angle.Lars'>", "<class 'sklearn.linear_model.least_angle.LassoLars'>", "<class 'sklearn.linear_model.bayes.BayesianRidge'>", "<class 'sklearn.linear_model.bayes.ARDRegression'>", "<class 'sklearn.linear_model.logistic.LogisticRegression'>", "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>", "<class 'sklearn.svm.classes.SVR'>", "<class 'sklearn.svm.classes.NuSVR'>", "<class 'sklearn.svm.classes.LinearSVR'>", "<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>")
    | ``scorer`` : instance of scikit-learn's make_scorer class
    |   ("<class 'sklearn.metrics.scorer._PredictScorer'>",)

:output tokens (senders):
    | ``dfy_predict`` : pandas dataframe
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
    |   ``<< host = sklearn    << function = cross_val_predict``
    |   ``<< track_header = True``
    |   ``<< n_jobs = 1``
    |   ``<< verbose = 0``
    |   ``<< fit_params = None``
    |   ``<< method = predict``
    |   ``<< pre_dispatch = 2 * n_jobs``
    |   ``<< estimator = required_required``
    |   ``<< groups = None``
    |   ``<< y = None``
    |   ``<< X = required_required``
    |   ``<< cv = None``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id estimator``
    |   ``>> id scorer``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict