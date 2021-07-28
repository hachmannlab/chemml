.. _ConvertFile:

ConvertFile
============

:task:
    | Enter

:subtask:
    | Convert

:host:
    | cheml

:function:
    | ConvertFile

:input tokens (receivers):
    | ``file_path`` : the path to the file that needs to be converted
    |   types: ("<type 'str'>", "<type 'dict'>")

:output tokens (senders):
    | ``converted_file_paths`` : list of paths to the converted files
    |   types: <type 'list'>


:required packages:
    | ChemML, 0.4.1
    | Babel, 2.3.4

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = ConvertFile``
    |   ``<< to_format = required_required``
    |   ``<< file_path = required_required``
    |   ``<< from_format = required_required``
    |   ``>> id file_path``
    |   ``>> id converted_file_paths``
    |
    .. note:: The documentation page for function parameters: https://openbabel.org/wiki/Babel
