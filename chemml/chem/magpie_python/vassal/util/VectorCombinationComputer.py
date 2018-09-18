import numpy as np
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve

class VectorCombinationComputer:
    """Class to find all combinations of 3 vectors that are shorter than a
    certain length.

    Attributes
    ----------
    input_vectors : array-like
        Vectors to be added.
    cutoff_dist_sq :  float
        Square of cutoff distance.
    include_zero : bool
        Whether to include the zero vector in the list.
    super_cells : list
        x, y, z coordinates of each vector shorter than cutoff.
    vectors : list
        All vectors shorter then cutoff.

    """
    def __init__(self, in_vectors, cutoff_distance, include_zero=True):
        """Function to create the tool to compute all combinations of input
        vectors shorter than cutoff distance.

        Parameters
        ----------
        in_vectors : array-like
            Vectors to be combined. Must be exactly 3 linearly-independent
            vectors.
        cutoff_distance :  float
            Desired cutoff distance.
        include_zero : bool
            Whether to include the zero vector in the list.

        Raises
        ------
        ValueError
            If length of in_vectors is not 3.
        """

        if len(in_vectors) != 3:
            raise ValueError("Expecting exactly three vectors.")

        # Vectors to be added.
        self.input_vectors = list(in_vectors)

        # Square of cutoff distance.
        self.cutoff_distance_sq = cutoff_distance ** 2

        # Whether to include the zero vector in the list.
        self.include_zero = include_zero

        # x, y, z coordinates of each vector shorter than cutoff.
        self.super_cells = []

        # All vectors shorter then cutoff.
        self.vectors = []

        self.get_all_vectors()

    def compute_vector(self, x):
        """Function to compute a certain combination of vectors stored in
        this array.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        output : array-like
            Combination.
        """
        i_v = np.array(self.input_vectors, dtype=float)
        return np.array([x.dot(y) for y in i_v.T])


    def get_all_vectors(self):
        """Function to compute all vectors shorter than cutoff distance.
        """

        # Create a matrix of basis vectors.
        basis = np.array(self.input_vectors, dtype=float).T

        # Create ability to invert it.
        det_basis = np.linalg.det(basis)


        if det_basis == 0 or det_basis < 1e-14:
            raise RuntimeError("Vectors are not linearly independent.")

        fac = lu_factor(basis)

        # Compute range of each variable.
        cutoff_distance = np.math.sqrt(self.cutoff_distance_sq)
        step_range = []

        for i in range(3):
            max_disp = 0.0
            for j in range(3):
                max_disp += np.dot(self.input_vectors[i], self.input_vectors[
                    j]) / norm(self.input_vectors[i])
            step_range.append(int(np.math.ceil(max_disp / cutoff_distance)) + 1)

        # Ensure that we have sufficient range to get the cutoff distance
        # away from the origin by checking that we have large enough range to
        #  access a point cutoff distance away along the direction of xy,
        # xz and yz cross products.
        for dir in range(3):
            point = np.cross(self.input_vectors[dir], self.input_vectors[(dir
                                    + 1) % 3])
            point = point * cutoff_distance / norm(point)
            sln = lu_solve(fac, point)
            step_range = [max(step_range[i], int(np.math.ceil(abs(sln[i]))))
                          for i in range(3)]

        # Create the initial vector.
        for x in range(-step_range[0], 1 + step_range[0]):
            for y in range(-step_range[1], 1 + step_range[1]):
                for z in range(-step_range[2], 1 + step_range[2]):
                    a = np.array([x, y, z])
                    l = self.compute_vector(a)
                    dist_sq = l[0] ** 2 + l[1] ** 2 +  l[2] ** 2
                    if dist_sq <= self.cutoff_distance_sq:
                        if not self.include_zero and x == 0 and y == 0 and z \
                                == 0:
                            continue
                        self.super_cells.append(a)
                        self.vectors.append(l)

    def get_vectors(self):
        """Function to get the list of all vectors shorter than cutoff.

        Returns
        -------
        output : array-like
            List of vectors.

        """
        return list(self.vectors)

    def get_supercell_coordinates(self):
        """Function to get the list of all image coordinates of vectors.

        Returns
        -------
        output : array-like
            List of supercell coordinates.
        """
        return list(self.super_cells)