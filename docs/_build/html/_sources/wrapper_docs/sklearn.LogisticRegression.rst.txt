.. _LogisticRegression:

LogisticRegression
===================

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | LogisticRegression

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's LogisticRegression class
    |   ("<class 'sklearn.linear_model.logistic.LogisticRegression'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's LogisticRegression class
    |   ("<class 'sklearn.linear_model.logistic.LogisticRegression'>",)
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
    |   ``<< host = sklearn    << function = LogisticRegression``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< warm_start = False``
    |   ``<< n_jobs = 1``
    |   ``<< intercept_scaling = 1``
    |   ``<< fit_intercept = True``
    |   ``<< max_iter = 100``
    |   ``<< class_weight = None``
    |   ``<< C = 1.0``
    |   ``<< penalty = l2``
    |   ``<< multi_class = ovr``
    |   ``<< random_state = None``
    |   ``<< dual = False``
    |   ``<< tol = 0.0001``
    |   ``<< solver = liblinear``
    |   ``<< verbose = 0``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html