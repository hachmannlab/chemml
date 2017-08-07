.. _RDKitFingerprint:

RDKitFingerprint
=================

:task:
    | Prepare Data

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | RDKitFingerprint

:input tokens (receivers):
    | ``molfile`` : string
    |   the path to the input molecules file/files

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features)
    |   output DataFrame of feature values
    | ``removed_rows`` : pandas DataFrame, shape(n_removed_rows, 1)
    |   A list of molecules that have been removed since they are not able te be imported and read by RDKit

:required parameters:
    | no required parameters for block and function
    |
    .. note:: The documentation for this function can be found here: :py:func:`cheml.chem.RDKFingerprint`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2
    .. _Pandas: http://pandas.pydata.org
    | RDKit_, version: 0.18.1
    .. _RDKit: http://www.rdkit.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = RDKitFingerprint``
    |   ``<< removeHs = True``
    |   ``<< FPtype = 'Morgan'``
    |   ``<< vector = 'bit'``
    |   ``<< nBits = 1024``
    |   ``<< radius  =  2``
    |   ``<< path  =  None``
    |   ``<< arguments  =  None``
    |   ``>> id molfile``
    |   ``>> df id``
    |   ``>> removed_rows id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.