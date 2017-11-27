.. _train_test_split:

train_test_split
=================

:task:
    | Prepare

:subtask:
    | split

:host:
    | sklearn

:function:
    | train_test_split

:input tokens (receivers):
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``dfx_test`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy_train`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy_test`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx_train`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = train_test_split``
    |   ``<< track_header = True``
    |   ``<< shuffle = True``
    |   ``<< train_size = None``
    |   ``<< random_state = None``
    |   ``<< test_size = 0.25``
    |   ``<< y = None``
    |   ``<< X = required_required``
    |   ``<< stratify = None``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id dfx_test``
    |   ``>> id dfy_train``
    |   ``>> id dfy_test``
    |   ``>> id dfx_train``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html