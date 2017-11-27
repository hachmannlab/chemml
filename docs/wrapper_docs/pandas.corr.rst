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
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | p, a
    | 0, .

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = corr``
    |   ``<< min_periods = 1``
    |   ``<< method = pearson``
    |   ``>> id df``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html