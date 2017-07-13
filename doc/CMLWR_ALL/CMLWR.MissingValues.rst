.. _MissingValues:

MissingValues
==============

:task:
    | Prepare Data

:subtask:
    | Preprocessor

:host:
    | cheml

:function:
    | MissingValues

:input tokens (receivers):
    | ``dfx`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame
    | ``dfy`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``dfx`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame
    | ``dfy`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame
    | ``api`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame

:required parameters:
    | func_method  ( required for the block: string - 'fit_transform' or 'transform')
    |
    .. note:: The documentation for this function can be found here_

    .. _here: :py:func:`cheml.preprocessing.missing_values`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2\n\n    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = MissingValues``
    |   ``<< strategy = "interpolate"``
    |   ``<< string_as_null  =  True``
    |   ``<< inf_as_null  =  True``
    |   ``<< missing_values  =  False``
    |   ``>> id dfx``
    |   ``>> id dfy``
    |   ``>> dfx id``
    |   ``>> dfy id``
    |   ``>> api id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.