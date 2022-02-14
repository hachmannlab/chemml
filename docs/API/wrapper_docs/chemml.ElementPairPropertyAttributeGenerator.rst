.. _ElementPairPropertyAttributeGenerator:

ElementPairPropertyAttributeGenerator
======================================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | chemml

:function:
    | ElementPairPropertyAttributeGenerator

:input tokens (receivers):
    | ``entries`` : list of entries from CompositionEntry class.
    |   types: ("<type 'list'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = chemml    << function = ElementPairPropertyAttributeGenerator``
    |   ``<< elemental_pair_properties = None``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
