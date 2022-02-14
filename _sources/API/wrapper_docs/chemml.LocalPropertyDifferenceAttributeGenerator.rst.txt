.. _LocalPropertyDifferenceAttributeGenerator:

LocalPropertyDifferenceAttributeGenerator
==========================================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | chemml

:function:
    | LocalPropertyDifferenceAttributeGenerator

:input tokens (receivers):
    | ``entries`` : list of entries from CrystalStructureEntry class.
    |   types: ("<type 'list'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = chemml    << function = LocalPropertyDifferenceAttributeGenerator``
    |   ``<< elemental_properties = required_required``
    |   ``<< shells = [1]``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
