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
    | ``elemental_properties`` : , (default:None)
    |   
    |   choose one of: []

:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = ElementalPropertyAttributeGenerator``
    |   ``<< elemental_properties = None``
    |   ``<< use_default_properties = True``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
