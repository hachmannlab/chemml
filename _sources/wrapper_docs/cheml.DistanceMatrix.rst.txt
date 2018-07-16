.. _DistanceMatrix:

DistanceMatrix
===============

:task:
    | Represent

:subtask:
    | distance matrix

:host:
    | cheml

:function:
    | DistanceMatrix

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = DistanceMatrix``
    |   ``<< norm_type = fro``
    |   ``<< nCores = 1``
    |   ``>> id df``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
