.. _PRDFAttributeGenerator:

PRDFAttributeGenerator
=======================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | cheml

:function:
    | PRDFAttributeGenerator

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
    |   ``<< host = cheml    << function = PRDFAttributeGenerator``
    |   ``<< cut_off_distance = 10.0``
    |   ``<< n_points = 20``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
