.. _LinearRegression:

LinearRegression
=================

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | LinearRegression

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's LinearRegression class
    |   ("<class 'sklearn.linear_model.base.LinearRegression'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's LinearRegression class
    |   ("<class 'sklearn.linear_model.base.LinearRegression'>",)
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
    |   ``<< host = sklearn    << function = LinearRegression``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = False``
    |   ``<< n_jobs = 1``
    |   ``<< fit_intercept = True``
    |   ``<< copy_X = True``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html