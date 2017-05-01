.. _Merge:

Merge
============

:task:
    | Input

:host:
    | cheml

:function:
    | Merge

:parameters:
    |
    .. note:: The documentation for this method can be found here: :py:func:`cheml.initialization.Merge`

:send tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features)
    |   feature and target values matrix

:receive tokens:
    | ``df1`` : pandas data frame, shape(n_samples, n_features)
    |   The left side input data matrix
    | ``df2`` : pandas data frame, shape(n_samples, n_features)
    |   The right side input data matrix

:requirements:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Input``
    |   ``<< host = cheml``
    |   ``<< function = Merge``
    |   ``>> id df1     >> id df2    >> df id``
