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
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``filepath`` : pandas dataframe
    |   ("<type 'str'>",)


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = SaveFile``
    |   ``<< index = False``
    |   ``<< record_time = False``
    |   ``<< format = csv``
    |   ``<< output_directory = None``
    |   ``<< header = True``
    |   ``<< filename = required_required``
    |   ``>> id df``
    |   ``>> id filepath``
    |
    .. note:: The documentation page for function parameters: 