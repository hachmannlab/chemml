.. _Split:

Split
=====

:task:
    | Prepare Data

:subtask:
    | basic operators

:host:
    | cheml

:function:
    | Split

:input tokens (receivers):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``df1`` : pandas DataFrame, shape(n_samples, n_features)
    |   selected columns of the input DataFrame
    | ``df2`` : pandas DataFrame, shape(n_samples, n_features)
    |   remaining (not-selected) columns of the input DataFrame

:required parameters:
    | selection 
    |
    .. note:: The documentation for this function can be found here: :py:func:`cheml.initialization.Split`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2
    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = Split``
    |   ``<< selection  =  1``
    |   ``>> id df``
    |   ``>> df1 id``
    |   ``>> df2 id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.