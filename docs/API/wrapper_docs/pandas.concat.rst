.. _concat:

concat
=======

:task:
    | Prepare

:subtask:
    | data manipulation

:host:
    | pandas

:function:
    | concat

:input tokens (receivers):
    | ``df1`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df3`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df2`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = concat``
    |   ``<< join = outer``
    |   ``<< verify_integrity = False``
    |   ``<< keys = None``
    |   ``<< levels = None``
    |   ``<< ignore_index = False``
    |   ``<< names = None``
    |   ``<< join_axes = None``
    |   ``<< copy = True``
    |   ``<< axis = 0``
    |   ``>> id df1``
    |   ``>> id df3``
    |   ``>> id df2``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html
