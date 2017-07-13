.. _SsveFile:

SaveFile
============

:task:
    | Store

:host:
    | cheml

:function:
    | SaveFile

:parameters:
    | filename
    | output_directory
    | record_time
    | format
    | index
    | header
    |
    .. note:: The documentation for parameters can be found here: :py:func:`cheml.initialization.SaveFile`

:send tokens:
    | ``filepath`` : string
    |   The saved file path

:receive tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features), requied
    |   input data frame

:requirements:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Store``
    |   ``<< host = cheml``
    |   ``<< function = SaveFile``
    |   ``<< filename = ''``
    |   ``<< output_directory = None``
    |   ``<< record_time = False``
    |   ``<< format = 'csv'``
    |   ``<< index = False``
    |   ``<< header = False``
    |   ``>> id df     >> filepath id``
