.. _read_table:

read_table
===========

:task:
    | Enter Data

:subtask:
    | input

:host:
    | pandas

:function:
    | read_table

:input tokens (receivers):
    | no receiver

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features)
    |   output DataFrame

:required parameters:
    | filepath_or_buffer 
    |
    .. note:: All the parameters of this function in the documentation page (and only those) are available. The documentation for this function can be found
        `here <http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.read_table.html`_.

:required packages:
    | Pandas_, version: 0.20.2
    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Enter Data``
    |   ``<< host = pandas    << function = read_table``
    |   ``<< filepath_or_buffer  =  ''``
    |   ``>> df id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.