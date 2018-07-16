.. _StoichiometricAttributeGenerator:

StoichiometricAttributeGenerator
=================================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | cheml

:function:
    | StoichiometricAttributeGenerator

:input tokens (receivers):
    | ``entries`` : list of entries from CompositionEntry class.
    |   types: ("<type 'list'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``use_default_norms`` : , (default:None)
    |   
    |   choose one of: []

:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = StoichiometricAttributeGenerator``
    |   ``<< use_default_norms = None``
    |   ``<< p_norms = None``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
