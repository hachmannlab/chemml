.. _load_cep_homo:

load_cep_homo
==============

:task:
    | Enter

:subtask:
    | datasets

:host:
    | cheml

:function:
    | load_cep_homo

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``smiles`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``homo`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = load_cep_homo``
    |   ``>> id smiles``
    |   ``>> id homo``
    |
    .. note:: The documentation page for function parameters: 
