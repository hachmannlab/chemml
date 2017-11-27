.. _BayesianRidge:

BayesianRidge
==============

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | BayesianRidge

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's BayesianRidge class
    |   ("<class 'sklearn.linear_model.bayes.BayesianRidge'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's BayesianRidge class
    |   ("<class 'sklearn.linear_model.bayes.BayesianRidge'>",)
    | ``dfy_predict`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit', 'predict', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = BayesianRidge``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = False``
    |   ``<< n_iter = 300``
    |   ``<< verbose = False``
    |   ``<< lambda_2 = 1e-06``
    |   ``<< fit_intercept = True``
    |   ``<< compute_score = False``
    |   ``<< alpha_2 = 1e-06``
    |   ``<< tol = 0.001``
    |   ``<< alpha_1 = 1e-06``
    |   ``<< copy_X = True``
    |   ``<< lambda_1 = 1e-06``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html