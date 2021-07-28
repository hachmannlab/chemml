.. _load_comp_energy:

load_comp_energy
=================

:task:
    | Enter

:subtask:
    | datasets

:host:
    | cheml

:function:
    | load_comp_energy

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``formation_energy`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``entries`` : list of entries from CompositionEntry class.
    |   types: ("<type 'list'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = load_comp_energy``
    |   ``>> id formation_energy``
    |   ``>> id entries``
    |
    .. note:: The documentation page for function parameters: 
