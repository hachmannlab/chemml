import numpy as np
from numpy.linalg import norm
from ..data.AtomImage import AtomImage
from ..geometry.Plane import Plane
from ..util.VectorCombinationComputer import VectorCombinationComputer

class PairDistanceAnalysis:
    """Class to compute the distances between each pairs of atom. Determines the
    distance between each of atoms (and any periodic image within a certain
    distance).

    Attributes
    ----------
    structure : Cell
        Link to structure being evaluated.
    cut_off_distance : float
        Cutoff distance. No pairs farther than this value are considered.
    supercells : list
        Periodic images to consider when searching for neighbors.
    lattice_vectors : list
        Lattice vectors corresponding to each image.

    """

    def __init__(self):
        """Function to instantiate the class.

        Parameters
        ----------
        structure : Cell
            Link to structure being evaluated.
        cut_off_distance : float
            Cutoff distance. No pairs farther than this value are considered.
        supercells : list
            Periodic images to consider when searching for neighbors.
        lattice_vectors : list
            Lattice vectors corresponding to each image.
        """

        # Cutoff distance. No pairs farther than this value are considered.
        self.cutoff_distance = 6.5

        # Periodic images to consider when searching for neighbors.
        self.supercells = []

        # Lattice vectors corresponding to each image.
        self.lattice_vectors = []

        # Link to structure being evaluated.
        self.structure = None

    def precompute(self):
        """Function to perform any kind of computations that should be
        performed only once.

        """

        lat_vectors = self.structure.get_lattice_vectors()
        # Goal: Compute all combinations of lattice vectors that are less
        # than the cutoff length plus the length of the longest vector.
        #
        # Find the largest distance between an atom and one of the vertices
        # of the unit cell. Computed using the Voronoi tessellation. This
        # point is the vertex corresponding to the faces of (origin, {a & b &
        #  c}).
        p0 = Plane(normal=lat_vectors[0], tolerance=1e-6, p= 0.5 * lat_vectors[0])
        p1 = Plane(normal=lat_vectors[1], tolerance=1e-6, p= 0.5 * lat_vectors[1])
        p2 = Plane(normal=lat_vectors[2], tolerance=1e-6, p= 0.5 * lat_vectors[2])
        x = Plane.intersection_3_planes(p0, p1, p2)
        max_image_dist = norm(x)

        # In order to create a list of vectors such the set of displacement
        # vectors from any atom in the structure to every image of another
        # atom is within the cutoff radius is a subset of the set of
        # displacement vectors created by adding the displacement vector of
        # the closest image of the second atom from the first atom to each
        # atom in the list of vectors, we need to find all vectors whose
        # length is shorter than max_dist + cutoff_distance.
        #
        # Rationale:
        # 1. The maximum distance between an atom and the closet neighbor
        # is no more than maxDist (defined above). Otherwise, the image is
        # not the closest image.
        # 2. If we are interested in finding all images of that second atom
        # that are within the cutoff distance of the first atom, the distance
        # between any of those images and the closest image of the second
        # atom can't be greater than the cutoff distance plus the distance
        # between that closest image and the first atom.
        # 3. Those images of the second atoms are located at integer
        # multiples of the lattice vectors. So, if we find all lattice
        # vectors shorter than cutoff_distance + longest possible distance
        # between an atom and the closest image of a second atom, we can find
        #  all images!

        # Compute those vectors.
        computer = VectorCombinationComputer(lat_vectors, max_image_dist +
                                             self.cutoff_distance)
        self.supercells = computer.get_supercell_coordinates()
        self.lattice_vectors = computer.get_vectors()

    def get_cutoff_distance(self):
        """Function to get the cutoff distance.

        Returns
        -------
        cutoff_distance : float
            Cutoff distance.

        """
        return self.cutoff_distance

    def set_cutoff_distance(self, d):
        """Function to set the cutoff distance.

        Parameters
        ----------
        d : float
            Desired cutoff distance.

        """
        self.cutoff_distance = d
        if self.structure is not None:
            self.precompute()

    def find_all_images(self, center_atom, neighbor_atom):
        """Function to find all images of one atom that are within the cutoff
        distance of another atom.

        Parameters
        ----------
        center_atom : int
            Index of atom at the center.
        neighbor_atom : int
            Neighboring atom index.

        Returns
        -------
        output : list
            All images of neighbor_atom that within a cutoff distance of
            center_atom, and their respective distances. List of tuples (
            neighboring atom, distance).

        """

        # Get the two atoms.
        center_pos = self.structure.get_atom(
            center_atom).get_position_cartesian()

        # Find all images.
        output = []
        closest_image = self.structure.get_minimum_distance(
            center=center_atom, neighbor=neighbor_atom)

        B_sub_A = closest_image.get_position()
        B_sub_A -= center_pos

        closest_supercell = closest_image.get_supercell()

        # Loop over each periodic image to find those within range.
        cutoff_distance_sq = self.cutoff_distance ** 2
        for img in range(len(self.supercells)):
            supercell_vec = self.lattice_vectors[img]
            new_pos = B_sub_A.copy() + supercell_vec
            dist = new_pos[0] ** 2 + new_pos[1] ** 2 + new_pos[2] ** 2

            if dist < cutoff_distance_sq and dist > 1e-8:
                ss = self.supercells[img].copy() + closest_supercell
                d = np.math.sqrt(dist)
                pos = closest_image.get_atom().get_position()
                output.append((AtomImage(closest_image.get_atom(), ss), d))

        return output

    def get_all_neighbors_of_atom(self, index):
        """Function to compute all neighbors of a certain atom.

        Parameters
        ----------
        index : int
            Index of atom being considered.

        Returns
        -------
        output : list
            List of tuples (neighboring atom, distance).

        """
        output = []
        for atom in self.structure.get_atoms():
            output += self.find_all_images(index, atom.get_id())
        return output

    def compute_PRDF(self, n_bin):
        """Function to compute the pair radial distribution function.

        Parameters
        ----------
        n_bin : int
            Number of bins in which to discretize PRDF (must be > 0)

        Returns
        -------
        output : array-like
            Pair radial distribution function between each type. Bin i,
            j,k is the density of bonds between types i and j that are between k
            * cutoff / nBin and (k + 1) * cutoff / nBin.

        Raises
        ------
        ValueError
            If n_bin is less than or equal to 0.
        """
        if n_bin <= 0:
            raise ValueError("Number of bins must be 0.")

        n_t = self.structure.n_types()
        n_a = self.structure.n_atoms()

        # Initialize arrays.
        output = np.zeros((n_t, n_t, n_bin))
        # Find all pairs within the cutoff distance.
        n_type = np.zeros(n_t, dtype=int)

        for i in range(n_a):
            for j in range(i, n_a):
                # Find images.
                images = self.find_all_images(i, j)
                i_type = self.structure.get_atom(i).get_type()
                n_type[i_type] += 1
                j_type = self.structure.get_atom(j).get_type()

                # For each image, assign it to bin.
                for img in images:
                    bin_ = int(np.math.floor(img[1] * n_bin /
                                             self.cutoff_distance))
                    if bin_ >= n_bin:
                        # Happens if dist equals cutoff.
                        continue
                    output[i_type][j_type][bin_] += 1
                    output[j_type][i_type][bin_] += 1

        # Normalizing data.
        bin_spacing = self.cutoff_distance / n_bin
        for b in range(n_bin):
            vol = 4.0 / 3.0 * ((b + 1) ** 3 - b ** 3) * bin_spacing ** 3 * \
                  np.math.pi
            for i in range(n_t):
                for j in range(n_t):
                    output[i][j][b] /= vol * n_type[i]

        return output

    def analyze_structure(self, s):
        """Function to analyze a specific structure. Once this completes,
        it is possible to retrieve results out of this object.

        Parameters
        ----------
        s : Cell
            Structure to be analyzed.

        """
        self.structure = s
        self.precompute()

    def recompute(self):
        """Function to recompute structural information.

        """
        self.precompute()
