import types
import numpy as np
import pandas as pd
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry
from ....models.regression.crystal.CoulombSineMatrixRegression import \
    CoulombSineMatrixRegression

class CoulombMatrixAttributeGenerator:
    """Class to compute attributes using the Coulomb Sine Matrix
    representation. Based on work by Faber et al. [1].

    Attributes
    ----------
    n_eigenvalues : int
        Maximum number of atoms to consider. Defines number of attributes.

    Notes
    -----
    This method works by computing an approximation for the Coulomb matrix
    that considers periodicity. Specifically, we use the Coulomb Sine matrix,
    which is described in detail in the Faber et al.[1]. For molecules,
    the Coulomb matrix is defined as

    .. math:: C_{i,j} &= Z_i^{2.4} & \text{if} i=j\\
                      &= Z_i Z_j / r_ij & \text{if} i != j

    The eigenvalues of this matrix are then used as attributes. In order to
    provided a fixed number of attributes, the first N attributes are defined
    to be the N eigenvalues from the Coulomb matrix. The remaining attributes
    are defined to be zero.

    The Coulomb Matrix attributes are dependant on unit cell choice.
    Please consider transforming your input crystal structures to the primitive
    cell before using these attributes.

    References
    ----------
    .. [1] F. Faber, A. Lindmaa, O. A. von Lilienfeld, and R. Armiento,
    "Crystal structure representations for machine learning models of
    formation energies," International Journal of Quantum Chemistry,
    vol. 115, no. 16, pp. 1094--1101, Apr. 2015.

    """

    def __init__(self):
        """Function to create instance and initialize fields.
        """

        # Maximum number of atoms to consider. Defines number of attributes.
        self.n_eigenvalues = 30

    def set_n_eigenvalues(self, x):
        """Function to set the number of eigenvalues used in representation.

        Parameters
        ----------
        x : int
            Desired number.

        """

        self.n_eigenvalues = x

    def generate_features(self, entries):
        """Function to generate features as mentioned in the class description.

        Parameters
        ----------
        entries : array-like
            Crystal structures for which features are to be generated. A list
            of CrystalStructureEntry's.

        Returns
        ----------
        features : DataFrame
            Features for the given entries. Pandas data frame containing the
            names and values of the descriptors.

        Raises
        ------
        ValueError
            If input is not of type list.
            If items in the list are not CrystalStructureEntry instances.

        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Insert header names here.
        feat_headers = ["CoulombMatrix_Eigenvalue" + str(i) for i in range(
            self.n_eigenvalues)]

        # Create tool to compute eigenvalues.
        tool = CoulombSineMatrixRegression()

        # Generate features.
        for entry in entries:
            eigenvalues = tool.compute_representation(entry.get_structure())
            tmp_array = np.zeros(self.n_eigenvalues, dtype=float)
            if len(eigenvalues) < self.n_eigenvalues:
                tmp_array[:len(eigenvalues)] = eigenvalues[:]
            else:
                tmp_array[:] = eigenvalues[:self.n_eigenvalues]
            feat_values.append(tmp_array)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features