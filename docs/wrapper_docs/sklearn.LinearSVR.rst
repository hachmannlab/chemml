.. _LinearSVR:

LinearSVR
==========

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | LinearSVR

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's LinearSVR class
    |   ("<class 'sklearn.svm.classes.LinearSVR'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's LinearSVR class
    |   ("<class 'sklearn.svm.classes.LinearSVR'>",)
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
    |   ``<< host = sklearn    << function = LinearSVR``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< loss = epsilon_insensitive``
    |   ``<< intercept_scaling = 1.0``
    |   ``<< fit_intercept = True``
    |   ``<< epsilon = 0.0``
    |   ``<< max_iter = 1000``
    |   ``<< C = 1.0``
    |   ``<< random_state = None``
    |   ``<< dual = True``
    |   ``<< tol = 0.0001``
    |   ``<< verbose = 0``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html