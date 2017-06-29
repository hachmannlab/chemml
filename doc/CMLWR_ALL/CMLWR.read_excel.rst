.. _read_excel:

read_excel
============

:task:
    | Enter Data

:subtask:
    | input

:host:
    | pandas

:function:
    | read_excel

:input tokens (receivers):
    | no receiver

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features)
    |   output DataFrame

:required parameters:
    | io
    |
    .. note:: All the parameters of this function in the documentation page (and only those) are available. The documentation for this function can be found here_.

    .. _here: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html#pandas.read_excel


:required packages:
    | Pandas_, version: 0.20.2

    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Enter Data``
    |   ``<< host = pandas    << function = read_excel``
    |   ``<< io  =  ''``
    |   ``>> df id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.