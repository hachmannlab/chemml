.. _SGDRegressor:

SGDRegressor
=============

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | SGDRegressor

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's SGDRegressor class
    |   ("<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's SGDRegressor class
    |   ("<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>",)
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
    |   ``<< host = sklearn    << function = SGDRegressor``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< warm_start = False``
    |   ``<< loss = squared_loss``
    |   ``<< eta0 = 0.01``
    |   ``<< verbose = 0``
    |   ``<< fit_intercept = True``
    |   ``<< l1_ratio = 0.15``
    |   ``<< average = False``
    |   ``<< n_iter = 5``
    |   ``<< penalty = l2``
    |   ``<< power_t = 0.25``
    |   ``<< alpha = 0.0001``
    |   ``<< random_state = None``
    |   ``<< epsilon = 0.1``
    |   ``<< shuffle = True``
    |   ``<< learning_rate = invscaling``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html