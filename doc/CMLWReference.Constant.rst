.. _Split:

Split
============

:task:
    | Prepare Data

:host:
    | cheml

:function:
    | Split

:parameters:
    | selection
    |
    .. note:: The documentation for parameters can be found here: :py:func:`cheml.initialization.Split`

:send tokens:
    | ``df1`` : pandas data frame, shape(n_samples, n_features)
    |   selected columns of the input data frame
    | ``df2`` : pandas data frame, shape(n_samples, n_features)
    |   remaining columns of the input data matrix

:receive tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features), requied
    |   input data frame

:requirements:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml``
    |   ``<< function = Split``
    |   ``<< selection = 1``
    |   ``>> id df    >> df1 id    >> df2 id``


