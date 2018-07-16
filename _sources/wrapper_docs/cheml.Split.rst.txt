.. _Split:

Split
======

:task:
    | Prepare

:subtask:
    | data manipulation

:host:
    | cheml

:function:
    | Split

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df1`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df2`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Split``
    |   ``<< selection = 1``
    |   ``>> id df``
    |   ``>> id df1``
    |   ``>> id df2``
    |
    .. note:: The documentation page for function parameters: 
