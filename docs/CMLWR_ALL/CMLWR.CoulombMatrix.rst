.. _CoulombMatrix:

CoulombMatrix
==============

:task:
    | Prepare Data

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | CoulombMatrix

:input tokens (receivers):
    | ``no receiver`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame

:required parameters:
    | no required parameters for block and function ()
    |
    .. note:: The documentation for this function can be found here_

    .. _here: :py:func:`cheml.chem.CoulombMatrix`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2\n\n    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = CoulombMatrix``
    |   ``<< CMtype = 'SC'``
    |   ``<< nPerm = 6``
    |   ``<< const = 1``
    |   ``>> df id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.