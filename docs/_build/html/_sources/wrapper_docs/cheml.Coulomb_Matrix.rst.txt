.. _Coulomb_Matrix:

Coulomb_Matrix
===============

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | Coulomb_Matrix

:input tokens (receivers):
    | ``molecules`` : the molecule numpy array or data frame
    |   ("<class 'pandas.core.frame.DataFrame'>", "<type 'numpy.ndarray'>", "<type 'dict'>")

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Coulomb_Matrix``
    |   ``<< const = 1``
    |   ``<< CMtype = SC``
    |   ``<< nPerm = 3``
    |   ``<< max_n_atoms = auto``
    |   ``>> id molecules``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 