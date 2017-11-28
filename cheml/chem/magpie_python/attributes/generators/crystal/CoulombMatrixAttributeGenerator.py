import types
import numpy as np
import pandas as pd
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from models.regression.crystal.CoulombSineMatrixRegression import \
    CoulombSineMatrixRegression

class CoulombMatrixAttributeGenerator:
    """
    Class to compute attributes using the Coulomb Sine Matrix representation.
    Based on work by Faber et al. http://doi.wiley.com/10.1002/qua.24917

    This method works by computing an approximation for the Coulomb matrix
    that considers periodicity. Specifically, we use the Coulomb Sine matrix,
    which is described in detail in the Faber <i>et al.</i>. For molecules,
    the Coulomb matrix is defined as

    C_{i,j} = Z_i^{2.4}             if i=j
              Z_iZ_j / r_ij         if i != j

    The eigenvalues of this matrix are then used as attributes. In order to
    provided a fixed number of attributes, the first N attributes are defined
    to be the N eigenvalues from the Coulomb matrix. The remaining attributes
    are defined to be zero.

    The Coulomb Matrix attributes are dependant on unit cell choice.
    Please consider transforming your input crystal structures to the primitive
    cell before using these attributes.
    """
    def __init__(self):
        """
        Function to create instance and initialize fields.
        """

        # Maximum number of atoms to consider. Defines number of attributes.
        self.n_eigenvalues = 30

    def set_n_eigenvalues(self, x):
        """
        Function to set the number of eigenvalues used in representation.
        :param x: Desired number.
        :return:
        """
        self.n_eigenvalues = x

    def generate_features(self, entries, verbose=False):
        """
        Function to generate features as mentioned in the class description.
        :param entries: A list of CrystalStructureEntry's.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.
        if (type(entries) is not types.ListType):
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
        if verbose:
            print features.head()
        return features