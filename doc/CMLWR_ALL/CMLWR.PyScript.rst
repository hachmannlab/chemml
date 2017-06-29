.. _PyScript:

PyScript
=========

:task:
    | Prepare Data

:subtask:
    | basic operators

:host:
    | cheml

:function:
    | PyScript

:input tokens (receivers):
    | ``df1`` : pandas DataFrame, shape(n_samples, n_features), optional
    |   input DataFrame
    | ``df2`` : pandas DataFrame, shape(n_samples, n_features), optional
    |   input DataFrame
    | ``api1`` : optional format, optional
    |   embedded for any python class
    | ``api2`` : optional format, optional
    |   embedded for any python class
    | ``var1`` : optional format, optional
    |   embedded for any other variable
    | ``var2`` : optional format, optional
    |   embedded for any other variable

:output tokens (senders):
    | ``df_out1`` : pandas DataFrame
    |   output DataFrame
    | ``df_out2`` : pandas DataFrame
    |   output DataFrame
    | ``api_out1`` : optional format
    |   any instance of a python class
    | ``api_out2`` : optional format
    |   any instance of a python class
    | ``var_out1`` : optional format
    |   any other variable
    | ``var_out2`` : optional format
    |   any other variable

:required parameters:
    | li
    |
    .. note:: The parameter name is simply the line number (li, i:int). These parameters will be sorted by their names to identify the order of python lines. The parametr value is a line of python code in string format.

:required packages:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = PyScript``
    |   ``<< l1  =  ""``
    |   ``>> id df1``
    |   ``>> id df2``
    |   ``>> id api1``
    |   ``>> id api2``
    |   ``>> id var1``
    |   ``>> id var2``
    |   ``>> df_out1 id``
    |   ``>> df_out2 id``
    |   ``>> api_out1 id``
    |   ``>> api_out2 id``
    |   ``>> var_out1 id``
    |   ``>> var_out2 id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.