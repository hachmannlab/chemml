.. _ReadTable:

ReadTable
============

:task:
    | Input

:host:
    | cheml

:function:
    | ReadTable

:parameters:
    | filepath
    | header
    | skiprows
    | skipcolumns
    |
    .. note:: The documentation for parameters can be found here: :py:func:`cheml.initialization.ReadTable`

:send tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features)
    |   feature and target values matrix

:receive tokens:
    | no tokens

:requirements:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Input``
    |   ``<< host = cheml``
    |   ``<< function = ReadTable``
    |   ``<< filepath = ''``
    |   ``<< header = 0``
    |   ``<< skiprows = 0``
    |   ``<< skipcolumns = 0``
    |   ``<< df id``

