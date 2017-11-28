.. _DistanceMatrix:

DistanceMatrix
===============

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | DistanceMatrix

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.1.0
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