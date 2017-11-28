.. _SVR:

SVR
====

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | SVR

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's SVR class
    |   ("<class 'sklearn.svm.classes.SVR'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's SVR class
    |   ("<class 'sklearn.svm.classes.SVR'>",)
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
    |   ``<< host = sklearn    << function = SVR``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< kernel = rbf``
    |   ``<< verbose = False``
    |   ``<< degree = 3``
    |   ``<< coef0 = 0.0``
    |   ``<< epsilon = 0.1``
    |   ``<< max_iter = -1``
    |   ``<< C = 1.0``
    |   ``<< tol = 0.001``
    |   ``<< cache_size = 200``
    |   ``<< shrinking = True``
    |   ``<< gamma = auto``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html