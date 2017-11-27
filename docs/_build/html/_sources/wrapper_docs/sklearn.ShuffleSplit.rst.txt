.. _ShuffleSplit:

ShuffleSplit
=============

:task:
    | Prepare

:subtask:
    | split

:host:
    | sklearn

:function:
    | ShuffleSplit

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's ShuffleSplit class
    |   ("<class 'sklearn.model_selection._split.ShuffleSplit'>",)
    | ``fold_gen`` : Generator of indices to split data into training and test set
    |   ("<type 'generator'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('split', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = ShuffleSplit``
    |   ``<< func_method = None``
    |   ``<< n_splits = 10``
    |   ``<< train_size = None``
    |   ``<< random_state = None``
    |   ``<< test_size = default``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id fold_gen``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit