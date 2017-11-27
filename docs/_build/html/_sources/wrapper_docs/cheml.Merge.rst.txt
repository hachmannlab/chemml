.. _Merge:

Merge
======

:task:
    | Prepare

:subtask:
    | basic operators

:host:
    | cheml

:function:
    | Merge

:input tokens (receivers):
    | ``df1`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df2`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Merge``
    |   ``>> id df1``
    |   ``>> id df2``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 