.. _CrystalStructureEntry:

CrystalStructureEntry
======================

:task:
    | Represent

:subtask:
    | inorganic input

:host:
    | cheml

:function:
    | CrystalStructureEntry

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``entries`` : list of entries from CrystalStructureEntry class.
    |   types: ("<type 'list'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = CrystalStructureEntry``
    |   ``<< directory_path = required_required``
    |   ``>> id entries``
    |
    .. note:: The documentation page for function parameters: 
