.. _Outliers:

Outliers
=========

:task:
    | Prepare

:subtask:
    | data cleaning

:host:
    | cheml

:function:
    | Outliers

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's Constant class
    |   types: ("<class 'cheml.preprocessing.purge.Outliers'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's Constant class
    |   types: ("<class 'cheml.preprocessing.purge.Outliers'>",)
    | ``removed_columns_`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit_transform', 'transform', None)

:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Outliers``
    |   ``<< func_method = None``
    |   ``<< m = 2.0``
    |   ``<< strategy = median``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id removed_columns_``
    |
    .. note:: The documentation page for function parameters: 
