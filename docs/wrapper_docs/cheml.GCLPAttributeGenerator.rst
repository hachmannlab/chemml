.. _GCLPAttributeGenerator:

GCLPAttributeGenerator
=======================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | cheml

:function:
    | GCLPAttributeGenerator

:input tokens (receivers):
    | ``energies`` : to be passed to the parameter energies
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``phases`` : to be passed to the parameter phases
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
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
    |   ``<< host = cheml    << function = GCLPAttributeGenerator``
    |   ``<< count_phases = None``
    |   ``<< energies = []``
    |   ``<< phases = []``
    |   ``>> id energies``
    |   ``>> id phases``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
