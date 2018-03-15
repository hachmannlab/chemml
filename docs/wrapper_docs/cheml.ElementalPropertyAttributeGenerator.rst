.. _ElementalPropertyAttributeGenerator:

ElementalPropertyAttributeGenerator
====================================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | cheml

:function:
    | ElementalPropertyAttributeGenerator

:input tokens (receivers):
    | ``entries`` : list of entries from CompositionEntry class.
    |   types: ("<type 'list'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``use_default_properties`` : , (default:None)
    |   
    |   choose one of: []

:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = ElementalPropertyAttributeGenerator``
    |   ``<< use_default_properties = None``
    |   ``<< elemental_properties = None``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
