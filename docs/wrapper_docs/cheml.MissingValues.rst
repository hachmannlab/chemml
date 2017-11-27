.. _MissingValues:

MissingValues
==============

:task:
    | Prepare

:subtask:
    | preprocessor

:host:
    | cheml

:function:
    | MissingValues

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's MissingValues class
    |   ("<class 'cheml.preprocessing.handle_missing.missing_values'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of ChemML's MissingValues class
    |   ("<class 'cheml.preprocessing.handle_missing.missing_values'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:None)
    |   
    |   choose one of: ('fit_transform', 'transform', None)

:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = MissingValues``
    |   ``<< func_method = None``
    |   ``<< strategy = ignore_row``
    |   ``<< inf_as_null = True``
    |   ``<< string_as_null = True``
    |   ``<< missing_values = False``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |
    .. note:: The documentation page for function parameters: 