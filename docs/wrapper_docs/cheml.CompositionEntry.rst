.. _CompositionEntry:

CompositionEntry
=================

:task:
    | Represent

:subtask:
    | inorganic input

:host:
    | cheml

:function:
    | CompositionEntry

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``entries`` : list of entries from CompositionEntry class.
    |   types: ("<type 'list'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = CompositionEntry``
    |   ``<< filepath = required_required``
    |   ``>> id entries``
    |
    .. note:: The documentation page for function parameters: 
