.. _MultiTaskLasso:

MultiTaskLasso
===============

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | MultiTaskLasso

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's MultiTaskLasso class
    |   ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's MultiTaskLasso class
    |   ("<class 'sklearn.linear_model.coordinate_descent.MultiTaskLasso'>",)
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
    |   ``<< host = sklearn    << function = MultiTaskLasso``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = False``
    |   ``<< warm_start = False``
    |   ``<< selection = cyclic``
    |   ``<< fit_intercept = True``
    |   ``<< max_iter = 1000``
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
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html