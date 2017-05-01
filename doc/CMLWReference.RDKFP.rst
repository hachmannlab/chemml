.. _RDKFP:

RDKitFingerprint
==================

:task:
    | DataRepresentation

:host:
    | cheml

:function:
    | RDKitFingerprint

:parameters:
    |
    .. note:: The documentation for this method can be found here: :py:func:`cheml.chem.RDKFP`

:send tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features)
    |   feature values matrix
    | ``removed_rows`` : list of row numbers (molecules' index)
    |   molecules that have been removed since they are not able te be imported and read by RDKit.

:receive tokens:
    | ``molfile`` : string
    |   The path to the input molecule files

:requirements:
    | RDKit_, version: 0.18.1

    .. _RDKit: http://www.rdkit.org

:input file view:
    | ``## DataRepresentation``
    |   ``<< host = cheml``
    |   ``<< function = RDKitFingerprint``
    |   ``<< removeHs = True``
    |   ``<< FPtype = Morgan``
    |   ``<< vector = bit``
    |   ``<< nBits = 1024``
    |   ``<< radius = 2``
    |   ``<< path = None``
    |   ``<< arguments = None``
    |   ``>> id molfile     >> df id   >> removed_rows id``
