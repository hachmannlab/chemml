.. _LassoLars:

LassoLars
==========

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | LassoLars

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's LassoLars class
    |   ("<class 'sklearn.linear_model.least_angle.LassoLars'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's LassoLars class
    |   ("<class 'sklearn.linear_model.least_angle.LassoLars'>",)
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
    |   ``<< host = sklearn    << function = LassoLars``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< normalize = True``
    |   ``<< fit_path = True``
    |   ``<< fit_intercept = True``
    |   ``<< positive = False``
    |   ``<< max_iter = 500``
    |   ``<< eps = 2.22044604925e-16``
    |   ``<< precompute = auto``
    |   ``<< copy_X = True``
    |   ``<< alpha = 1.0``
    |   ``<< verbose = False``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html