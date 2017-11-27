.. _KernelRidge:

KernelRidge
============

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | KernelRidge

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's KernelRidge class
    |   ("<class 'sklearn.kernel_ridge.KernelRidge'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's KernelRidge class
    |   ("<class 'sklearn.kernel_ridge.KernelRidge'>",)
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
    |   ``<< host = sklearn    << function = KernelRidge``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< kernel = linear``
    |   ``<< degree = 3``
    |   ``<< kernel_params = None``
    |   ``<< alpha = 1``
    |   ``<< coef0 = 1``
    |   ``<< gamma = None``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html