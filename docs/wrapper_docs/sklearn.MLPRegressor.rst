.. _MLPRegressor:

MLPRegressor
=============

:task:
    | Model

:subtask:
    | regression

:host:
    | sklearn

:function:
    | MLPRegressor

:input tokens (receivers):
    | ``api`` : instance of scikit-learn's MLPRegressor class
    |   ("<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's MLPRegressor class
    |   ("<class 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>",)
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
    |   ``<< host = sklearn    << function = MLPRegressor``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< shuffle = True``
    |   ``<< verbose = False``
    |   ``<< random_state = None``
    |   ``<< tol = 0.0001``
    |   ``<< validation_fraction = 0.1``
    |   ``<< learning_rate = constant``
    |   ``<< momentum = 0.9``
    |   ``<< warm_start = False``
    |   ``<< epsilon = 1e-08``
    |   ``<< activation = relu``
    |   ``<< max_iter = 200``
    |   ``<< batch_size = auto``
    |   ``<< alpha = 0.0001``
    |   ``<< early_stopping = False``
    |   ``<< beta_1 = 0.9``
    |   ``<< beta_2 = 0.999``
    |   ``<< nesterovs_momentum = True``
    |   ``<< hidden_layer_sizes = (100,)``
    |   ``<< solver = adam``
    |   ``<< power_t = 0.5``
    |   ``<< learning_rate_init = 0.001``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor