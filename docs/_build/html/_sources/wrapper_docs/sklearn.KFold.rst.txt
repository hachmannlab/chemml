.. _KFold:

KFold
======

:task:
    | Prepare

:subtask:
    | split

:host:
    | sklearn

:function:
    | KFold

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's KFold class
    |   ("<class 'sklearn.model_selection._split.KFold'>",)
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
    |   ``<< host = sklearn    << function = KFold``
    |   ``<< func_method = None``
    |   ``<< random_state = None``
    |   ``<< shuffle = False``
    |   ``<< n_splits = 3``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id fold_gen``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html