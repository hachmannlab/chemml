.. _Lasso:

Lasso
======

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | Lasso

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's Lasso class
    |   ("<class 'sklearn.linear_model.coordinate_descent.Lasso'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's Lasso class
    |   ("<class 'sklearn.linear_model.coordinate_descent.Lasso'>",)
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
    |   ``<< host = sklearn    << function = Lasso``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = False``
    |   ``<< warm_start = False``
    |   ``<< selection = cyclic``
    |   ``<< fit_intercept = True``
    |   ``<< positive = False``
    |   ``<< max_iter = 1000``
    |   ``<< precompute = False``
    |   ``<< random_state = None``
    |   ``<< tol = 0.0001``
    |   ``<< copy_X = True``
    |   ``<< alpha = 1.0``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html