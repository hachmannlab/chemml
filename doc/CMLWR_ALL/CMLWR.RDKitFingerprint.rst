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
    | ``molfile`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame
    | ``removed_rows`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame

:required parameters:
    | no required parameters for block and function ()
    |
    .. note:: The documentation for this function can be found here_

    .. _here: :py:func:`cheml.chem.RDKFingerprint`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2\n\n    .. _Pandas: http://pandas.pydata.org
    | RDKit_, version: 0.18.1\n\n    .. _RDKit: http://www.rdkit.org

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = RDKitFingerprint``
    |   ``<< removeHs = True``
    |   ``<< FPtype = 'Morgan'``
    |   ``<< vector = 'bit'``
    |   ``<< nBits = 1024``
    |   ``<< radius  =  2``
    |   ``<< path  =  None``
    |   ``<< arguments  =  []``
    |   ``>> id molfile``
    |   ``>> df id``
    |   ``>> removed_rows id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.