import numpy as np
import math
from ....data.materials.util.LookUpData import LookUpData

class CoulombSineMatrixRegression:
    """Class to perform regression based on the Coulomb Sine Matrix approach of
    Faber et al. [1].

    Attributes
    ----------
    sigma : float
        Normalization term in kernel function.

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

        # Normalization term in kernel function.
        self.sigma = 1

    def set_sigma(self, s):
        """Function to set the normalization parameter in the kernel function.

        Parameters
        ----------
        s : float
            Desired normalization parameter.
        """
        self.sigma = s

    def compute_similarity(self, structure1, structure2):
        """Function to compute similarity between two crystal structures.

        Parameters
        ----------
        structure1 : array-like
            Representation of structure #1.
        structure2 : array-like
            Representation of structure #2.

        Returns
        -------
        output : float
            Similarity between the two structures.

        """
        e1 = structure1.copy()
        e2 = structure2.copy()

        # Determine which is bigger.
        flag = len(e1) > len(e2)
        len_diff = len(e1) - len(e2)
        if flag:
            e2 = np.lib.pad(e2, (0, len_diff), 'constant', constant_values=(0))
            diff = e1 - e2
        else:
            e1 = np.lib.pad(e1, (0, -len_diff), 'constant', constant_values=(
                0))
            diff = e2 - e1

        # Compute the L1 distance.
        dist = sum(np.abs(diff))

        # Compute the Laplacian.
        return math.exp(-1 * dist / self.sigma)

    def compute_representation(self, structure):
        """Function to compute the representation of the crystal structure.

        Parameters
        ----------
        structure : CrystalStructureEntry
            Crystal structure.

        Returns
        -------
        output : array-like
            Representation of the structure.

        """

        # First generate the Coulomb matrix.
        matrix = self.compute_coulomb_matrix(structure)

        # Compute the eigenvalues.
        w, v = np.linalg.eig(matrix)
        return w

    def compute_coulomb_matrix(self, structure):
        """Function to compute the Coulomb sine matrix. Equation 24 of the
        paper describing this method.

        Parameters
        ----------
        structure : CrystalStructureEntry
            Structure to be evaluated.

        Returns
        -------
        output : array-like
            Coulomb sine matrix.

        Raises
        ------
        Exception
            If element is not found.
        """

        # Get basis vectors.
        basis = structure.get_basis()

        # Get reciprocal basis vectors.
        basis_inverse = structure.get_inverse_basis()

        # Create output matrix.
        n_atoms = structure.n_atoms()
        n_types = structure.n_types()
        output = np.zeros((n_atoms, n_atoms), dtype=float)

        # Get the positions for each atom.
        pos = np.array([structure.get_atom(a).get_position_cartesian() for a
                        in range(n_atoms)])

        # Get the atomic number of each element.
        type_z = np.zeros(n_types, dtype=int)
        for i in range(n_types):
            if structure.get_type_name(i) in LookUpData.element_names:
                type_z[i] = 1 + LookUpData.element_names.index(
                            structure.get_type_name(i))
            else:
                raise Exception("No such element: "+structure.get_type_name(i))

        # Compute all terms.
        for r1 in range(n_atoms):
            output[r1, r1]  = 0.5 * (type_z[structure.get_atom(r1).get_type(
                                )]) ** 2.4
            for r2 in range(r1 + 1, n_atoms):
                # Displacement between the two atoms.
                disp = pos[r1] - pos[r2]

                # Convert to fractional coordinates.
                disp = np.matmul(basis_inverse, disp)

                # Multiply by pi and compute sin^2 of each element.
                disp = np.sin(disp * np.math.pi) ** 2

                # Multiply by basis vectors.
                disp = np.matmul(basis, disp)

                # Get the result.
                res = 1.0 / np.linalg.norm(disp)
                res *= type_z[structure.get_atom(r1).get_type()] * type_z[
                    structure.get_atom(r2).get_type()]
                output[r1, r2] = output[r2, r1] = res

        return output
