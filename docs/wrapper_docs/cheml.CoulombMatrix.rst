.. _CoulombMatrix:

CoulombMatrix
==============

:task:
    | Represent

:subtask:
    | molecular descriptors

:host:
    | cheml

:function:
    | CoulombMatrix

:input tokens (receivers):
    | ``molecules`` : the molecule numpy array or data frame
    |   types: ("<class 'pandas.core.frame.DataFrame'>", "<type 'numpy.ndarray'>", "<type 'dict'>")

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = CoulombMatrix``
    |   ``<< const = 1``
    |   ``<< CMtype = SC``
    |   ``<< nPerm = 3``
    |   ``<< max_n_atoms = auto``
    |   ``>> id molecules``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
