.. _Ridge:

Ridge
======

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | Ridge

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's Ridge class
    |   ("<class 'sklearn.linear_model.ridge.Ridge'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's Ridge class
    |   ("<class 'sklearn.linear_model.ridge.Ridge'>",)
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
    |   ``<< host = sklearn    << function = Ridge``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = False``
    |   ``<< fit_intercept = True``
    |   ``<< max_iter = None``
    |   ``<< random_state = None``
    |   ``<< tol = 0.001``
    |   ``<< copy_X = True``
    |   ``<< alpha = 1.0``
    |   ``<< solver = auto``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html