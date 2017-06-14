.. _read_table:

read_table
============

:task:
    | Enter Data

:host:
    | pandas

:function:
    | read_table

:parameters:
    | all the parameters of this function in the documentation page (and only those) are available.

:send tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features)
    |   feature and target values matrix

:receive tokens:
    | no tokens

:parameters documentation:
    pandas.read_table: find it here_

    .. _here: http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.read_table.html

:requirements:
    Pandas_, version: 0.20.2

    .. _Pandas: http://pandas.pydata.org

:input file view:
    |   ``## Enter Data``
    |            ``<< host = pandas``            ``<< function = read_table``

    |            ``<< filepath_or_buffer = ''``
    |            ``<< ...``

    |            ``>> df id``

    .. note:: The rest of parameters are not required, but they still can be passed to the function.

