.. _mlp_hogwild:

mlp_hogwild
============

:task:
    | Model

:subtask:
    | regression

:host:
    | cheml

:function:
    | mlp_hogwild

:input tokens (receivers):
    | ``api`` : instance of ChemML's mlp_hogwild class
    |   ("<class 'cheml.nn.nn_psgd.mlp_hogwild'>",)
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of ChemML's mlp_hogwild class
    |   ("<class 'cheml.nn.nn_psgd.mlp_hogwild'>",)
    | ``dfy_predict`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit', 'predict', None)

:required packages:
    | ChemML, 0.1.0
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = mlp_hogwild``
    |   ``<< func_method = None``
    |   ``<< rms_decay = 0.9``
    |   ``<< learn_rate = 0.001``
    |   ``<< nneurons = *required``
    |   ``<< input_act_funcs = *required``
    |   ``<< batch_size = 256``
    |   ``<< n_epochs = 10000``
    |   ``<< validation_size = 0.2``
    |   ``<< print_level = 1``
    |   ``<< n_hist = 20``
    |   ``<< threshold = 0.1``
    |   ``<< model = None``
    |   ``<< n_check = 50``
    |   ``<< n_cores = 1``
    |   ``>> id api``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id dfy_predict``
    |
    .. note:: The documentation page for function parameters: 