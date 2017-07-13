.. _NN_PSGD:

NN_PSGD
========

:task:
    | Define Model

:subtask:
    | regression

:host:
    | cheml

:function:
    | NN_PSGD

:input tokens (receivers):
    | ``dfx_train`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame
    | ``dfy_train`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame
    | ``dfx_test`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``dfy_train_pred`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame
    | ``dfy_test_pred`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame
    | ``model`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame

:required parameters:
    | nneurons  ( required for the function)
    | input_act_funcs  ( required for the function)
    |
    .. note:: The documentation for this function can be found here_

    .. _here: :py:func:`cheml.nn.nn_psgd`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2\n\n    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Define Model``
    |   ``<< host = cheml    << function = NN_PSGD``
    |   ``<< nneurons = []``
    |   ``<< input_act_funcs = []``
    |   ``<< validation_size = 0.2``
    |   ``<< learn_rate = 0.001``
    |   ``<< rms_decay = 0.9``
    |   ``<< n_epochs = 10000``
    |   ``<< batch_size = 256``
    |   ``<< n_cores = 1``
    |   ``<< n_hist = 20``
    |   ``<< n_check = 50``
    |   ``<< threshold = 0.1``
    |   ``<< print_level = 1``
    |   ``>> id dfx_train``
    |   ``>> id dfy_train``
    |   ``>> id dfx_test``
    |   ``>> dfy_train_pred id``
    |   ``>> dfy_test_pred id``
    |   ``>> model id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.