.. _load_organic_density:

load_organic_density
=====================

:task:
    | Enter

:subtask:
    | datasets

:host:
    | cheml

:function:
    | load_organic_density

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``smiles`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``features`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``density`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = load_organic_density``
    |   ``>> id smiles``
    |   ``>> id features``
    |   ``>> id density``
    |
    .. note:: The documentation page for function parameters: 
