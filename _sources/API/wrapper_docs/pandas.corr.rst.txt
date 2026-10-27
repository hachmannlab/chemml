.. _corr:

corr
=====

:task:
    | Search

:subtask:
    | evaluate

:host:
    | pandas

:function:
    | corr

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = corr``
    |   ``<< min_periods = 1``
    |   ``<< method = pearson``
    |   ``>> id df``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html
