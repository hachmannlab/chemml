.. _Lars:

Lars
=====

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | Lars

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's Lars class
    |   ("<class 'sklearn.linear_model.least_angle.Lars'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's Lars class
    |   ("<class 'sklearn.linear_model.least_angle.Lars'>",)
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
    |   ``<< host = sklearn    << function = Lars``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< n_nonzero_coefs = 500``
    |   ``<< normalize = True``
    |   ``<< fit_path = True``
    |   ``<< fit_intercept = True``
    |   ``<< positive = False``
    |   ``<< eps = 2.22044604925e-16``
    |   ``<< precompute = auto``
    |   ``<< copy_X = True``
    |   ``<< verbose = False``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html