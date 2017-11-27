.. _Constant:

Constant
=========

:task:
    | Prepare

:subtask:
    | preprocessor

:host:
    | cheml

:function:
    | Constant

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's Constant class
    |   ("<class 'cheml.preprocessing.purge.Constant'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's Constant class
    |   ("<class 'cheml.preprocessing.purge.Constant'>",)
    | ``removed_columns_`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit_transform', 'transform', None)

:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Constant``
    |   ``<< func_method = None``
    |   ``<< selection = 1``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id removed_columns_``
    |
    .. note:: The documentation page for function parameters: 