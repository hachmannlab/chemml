.. _SaveFile:

SaveFile
=========

:task:
    | Store

:subtask:
    | file

:host:
    | cheml

:function:
    | SaveFile

:input tokens (receivers):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``filepath`` : string
    |   the saved file path

:required parameters:
    | filename : required for the function
    |
    .. note:: The documentation for this function can be found here: :py:func:`cheml.initialization.SaveFile`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2
    .. _Pandas: http://pandas.pydata.org

:input file view:
    | ``## Store``
    |   ``<< host = cheml    << function = SaveFile``
    |   ``<< filename = ''``
    |   ``<< output_directory  =  None``
    |   ``<< record_time  =  False``
    |   ``<< format  = 'csv'``
    |   ``<< index  =  False``
    |   ``<< header  =  True``
    |   ``>> id df``
    |   ``>> filepath id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.