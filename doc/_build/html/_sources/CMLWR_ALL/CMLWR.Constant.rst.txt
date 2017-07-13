.. _Constant:

Constant
=========

:task:
    | Prepare Data

:subtask:
    | basic operators

:host:
    | *cheml*

:function:
    | *Constant*

:input tokens (>> receivers):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders >>):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame, selected columns of the input data frame
    | ``removed_columns_`` : pandas data frame, shape(n_removed_columns, 1)
    |   indices of removed columns

:required parameters:
    | no required parameters for block and function
    |
    .. note:: The documentation for this function can be found here: :py:func:`cheml.preprocessing.Constant`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2

    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = Constant``
    |   ``>> id df``
    |   ``>> df id``
    |   ``>> removed_columns_ id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.