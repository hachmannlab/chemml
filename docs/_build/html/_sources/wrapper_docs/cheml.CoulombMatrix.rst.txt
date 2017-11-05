.. _CoulombMatrix:

CoulombMatrix
==============

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | CoulombMatrix

:input tokens (receivers):
    | ``molfile`` : the molecule file path
    |   ("<type 'str'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``## ``
    |   ``<< host = cheml    << function = CoulombMatrix``
    |   ``<< const = 1``
    |   ``<< molfile = * required``
    |   ``<< CMtype = SC``
    |   ``<< skip_lines = [2, 0]``
    |   ``<< nPerm = 6``
    |   ``<< arguments = []``
    |   ``<< reader = auto``
    |   ``<< path = None``
    |   ``>> id molfile``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 