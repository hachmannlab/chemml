.. _NuSVR:

NuSVR
======

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | NuSVR

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's NuSVR class
    |   ("<class 'sklearn.svm.classes.NuSVR'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's NuSVR class
    |   ("<class 'sklearn.svm.classes.NuSVR'>",)
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
    |   ``<< host = sklearn    << function = NuSVR``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< kernel = rbf``
    |   ``<< verbose = False``
    |   ``<< degree = 3``
    |   ``<< coef0 = 0.0``
    |   ``<< max_iter = -1``
    |   ``<< C = 1.0``
    |   ``<< tol = 0.001``
    |   ``<< cache_size = 200``
    |   ``<< shrinking = True``
    |   ``<< nu = 0.5``
    |   ``<< gamma = auto``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html