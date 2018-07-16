.. _load_xyz_polarizability:

load_xyz_polarizability
========================

:task:
    | Enter

:subtask:
    | datasets

:host:
    | cheml

:function:
    | load_xyz_polarizability

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``coordinates`` : dictionary of molecules represented by their xyz coordinates and atomic numbers
    |   types: ("<type 'dict'>",)
    | ``polarizability`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = load_xyz_polarizability``
    |   ``>> id coordinates``
    |   ``>> id polarizability``
    |
    .. note:: The documentation page for function parameters: 
