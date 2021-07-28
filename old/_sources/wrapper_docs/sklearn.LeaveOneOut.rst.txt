.. _LeaveOneOut:

LeaveOneOut
============

:task:
    | Prepare

:subtask:
    | split

:host:
    | sklearn

:function:
    | LeaveOneOut

:input tokens (receivers):
    | ``dfx`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``api`` : instance of scikit-learn's LeaveOneOut class
    |   types: ("<class 'sklearn.model_selection._split.LeaveOneOut'>",)
    | ``fold_gen`` : Generator of indices to split data into training and test set
    |   types: ("<type 'generator'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('split', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = LeaveOneOut``
    |   ``<< func_method = None``
    |   ``>> id dfx``
    |   ``>> id api``
    |   ``>> id fold_gen``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
