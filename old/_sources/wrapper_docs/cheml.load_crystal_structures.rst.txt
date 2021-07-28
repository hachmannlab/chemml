.. _load_crystal_structures:

load_crystal_structures
========================

:task:
    | Enter

:subtask:
    | datasets

:host:
    | cheml

:function:
    | load_crystal_structures

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
    |   ``<< host = cheml    << function = load_crystal_structures``
    |   ``>> id entries``
    |
    .. note:: The documentation page for function parameters: 
