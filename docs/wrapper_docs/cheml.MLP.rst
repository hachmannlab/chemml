.. _MLP:

MLP
====

:task:
    | Model

:subtask:
    | regression

:host:
    | cheml

:function:
    | MLP

:input tokens (receivers):
    | ``api`` : instance of cheml.nn.keras.MLP class
    |   types: ("<class 'cheml.nn.keras.mlp.MLP'>",)
    | ``dfy`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of cheml.nn.keras.MLP class
    |   types: ("<class 'cheml.nn.keras.mlp.MLP'>",)
    | ``dfy_predict`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit', 'predict', None)

:required packages:
    | ChemML, 0.4.1
    | keras, 2.1.2
    | tensorflow, 1.4.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = MLP``
    |   ``<< func_method = None``
    |   ``<< nhidden = 1``
    |   ``<< loss = mean_squared_error``
    |   ``<< learning_rate = 0.01``
    |   ``<< layer_config_file = None``
    |   ``<< batch_size = 100``
    |   ``<< lr_decay = 0.0``
    |   ``<< regression = True``
    |   ``<< nclasses = None``
    |   ``<< activations = None``
    |   ``<< opt_config_file = None``
    |   ``<< nepochs = 100``
    |   ``<< nneurons = 100``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: 
