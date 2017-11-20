.. _Merge:

Merge
======

:task:
    | Prepare Data

:subtask:
    | basic operators

:host:
    | cheml

:function:
    | Merge

:input tokens (receivers):
    | ``df1`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame
    | ``df2`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features)
    |   output DataFrame

:required parameters:
    | no required parameters for block and function
    |
    .. note:: The documentation for this function can be found here: :py:func:`cheml.initialization.Merge`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2

    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = Merge``
    |   ``>> id df1``
    |   ``>> id df2``
    |   ``>> df id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.