.. _RDKitFingerprint:

RDKitFingerprint
=================

:task:
    | Represent

:subtask:
    | molecular descriptors

:host:
    | cheml

:function:
    | RDKitFingerprint

:input tokens (receivers):
    | ``molfile`` : the molecule file path
    |   types: ("<type 'str'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``removed_rows`` : output variable, of any format
    |   types: ()


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | RDKit, 2016.03.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = RDKitFingerprint``
    |   ``<< nBits = 1024``
    |   ``<< molfile = required_required``
    |   ``<< removeHs = True``
    |   ``<< vector = bit``
    |   ``<< radius = 2``
    |   ``<< arguments = []``
    |   ``<< path = None``
    |   ``<< FPtype = Morgan``
    |   ``>> id molfile``
    |   ``>> id df``
    |   ``>> id removed_rows``
    |
    .. note:: The documentation page for function parameters: 
