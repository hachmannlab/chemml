# py2 and py3 compatible
from future.utils import iteritems

from numpy.linalg import norm
import numpy as np
from .voronoi.VoronoiTessellationCalculator import \
    VoronoiTessellationCalculator

class VoronoiCellBasedAnalysis:
    """Class to perform structure analysis based on the Voronoi tessellation
    method.

    Attributes
    ----------
    structure : Cell
        Link to structure being evaluated.
    cells : array-like
        Voronoi cells for each atom in structure.
    radical : bool
        Whether to use a radical Voronoi tessellation.

    """

    def __init__(self, radical=None, old_tessellation=None,
                 new_structure=None):
        """Function to initialize a Voronoi cell analyzer.
        Create a new instance of the analysis toolkit based on a structure
        that has an identical tessellation. This occurs when the two
        structures have identical lattice parameters and atomic positions,
        but different identities of elements on those sites.

        Parameters
        ----------
        new_structure : Cell
            New structure.
        radical : bool
            Whether to use a radical Voronoi tessellation.
        old_tessellation : VoronoiCellBasedAnalysis
            Tessellation of original structure.

        Raises
        ------
        Exception
            If structures have different number of atoms.
            If basis vectors of structures are different.
            If any atom position is different
        """

        # Link to structure being evaluated.
        self.structure = None

        # Voronoi cells for each atom in structure.
        self.cells = None

        # Whether to use a radical Voronoi tessellation.
        self.radical = False

        if radical is not None:
            self.radical = radical
        else:
            # Check whether structures are similar.
            old_structure = old_tessellation.structure
            if old_structure.n_atoms() != new_structure.n_atoms():
                raise Exception("Structures have different number of atoms.")
            if norm(old_structure.get_basis() - new_structure.get_basis()) > \
                    1e-6:
                raise Exception("Basis vectors of structures are different.")
            for a in range(old_structure.n_atoms()):
                old_atom_pos = old_structure.get_atom(a).get_position()
                new_atom_pos = new_structure.get_atom(a).get_position()
                if not np.allclose(old_atom_pos, new_atom_pos, rtol=1e-6):
                    raise Exception("Atom # {} position is different".format(
                        a))

            # Create new instance.
            self.cells = old_tessellation.cells
            self.structure = new_structure
            self.radical = old_tessellation.radical

    def precompute(self):
        """Function to perform any kind of computations that should be
        performed only once.

        Raises
        ------
        Exception
            If geometry is invalid.
        """
        self.cells = VoronoiTessellationCalculator.compute(self.structure,
                                                           self.radical)
        for cell in self.cells:
            if not cell.geometry_is_valid():
                self.cells = None
                raise Exception("Invalid geometry.")

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

    def tessellation_is_converged(self):
        """Function to check whether the tessellation of this structure was
        successful.

        """
        return self.cells is not None

    def get_effective_coordination_numbers(self):
        """Function to get the effective coordination number.
        Defined as N_eff = 1 / sum[(f_i / SA_i)^2] where f_i is the area of
        face i.

        Parameters
        ----------

        Returns
        -------
        output : list
            Effective coordination number for each atom.
        """

        # Compute the coordination number.
        output = [1.0 / sum([(f_i.get_area() / cell.get_surface_area()) ** 2
                    for f_i in cell.get_faces()]) for cell in self.cells]
        return output

    def face_count_average(self):
        """Function to get the average number of faces on all cells.

        Returns
        -------
        output : float
            Average number.

        """
        return np.average([cell.n_faces() for cell in self.cells])

    def face_count_variance(self):
        """Function to get the variance in face count.

        Returns
        -------
        output : float
            Mean absolute deviation in coordination number.
        """
        avg = self.face_count_average()
        return np.mean([abs(cell.n_faces() - avg) for cell in self.cells])

    def face_count_minimum(self):
        """Function to get the minimum face count of all cells.

        Returns
        -------
        output : float
            Minimum.
        """
        return min([cell.n_faces() for cell in self.cells])

    def face_count_maximum(self):
        """Function to get the maximum face count of all cells.

        Parameters
        ----------

        Returns
        -------
        output : float
            Maximum.
        """
        return max([cell.n_faces() for cell in self.cells])

    def get_unique_polyhedron_shapes(self):
        """Function to get a list of all unique polyhedron shapes.

        Returns
        -------
        output : list
            Set of polyhedron shapes.
        """
        output = []
        for cell in self.cells:
            x = cell.get_polyhedron_shape()
            if x not in output:
                output.append(x)
        return output

    def volume_variance(self):
        """Function to compute the mean absolute deviation in the volume of each
        cell.

        Returns
        -------
        output : float
            Variance of cell volume fraction of all cells.
        """
        avg_volume = self.structure.volume() / self.structure.n_atoms()
        return np.mean([abs(self.cells[i].get_volume() - avg_volume) for i
                           in range(self.structure.n_atoms())])

    def volume_fraction_minimum(self):
        """Function to compute the fraction of cell volume occupied by the
        smallest Voronoi cell.

        Returns
        -------
        output : float
            Volume fraction of the smallest cell.
        """
        return min([cell.get_volume() for cell in self.cells]) / \
               self.structure.volume()

    def volume_fraction_maximum(self):
        """Function to compute the fraction of cell volume occupied by the
        largest Voronoi cell.

        Returns
        -------
        output : float
            Volume fraction of the largest cell.

        """
        return max([cell.get_volume() for cell in self.cells]) / \
               self.structure.volume()

    def max_packing_efficiency(self):
        """Function to compute the maximum packing efficiency assuming atoms are
        hard spheres.
        Algorithm:
        1. For each cell in the Voronoi tessellation of this cell, determine the
        minimum distance from the cell center to a face. This marks the maximum
        atomic radius.
        2. Compute the total volume represented by those maximally-sized atoms.
        3. Packing efficiency is atom volume divided by cell volume.

        Returns
        -------
        output : float
            Maximum packing efficiency.

        """
        atom_vol = sum([min_dist ** 3 for min_dist in [min([
            face.get_face_distance() for face in cell.get_faces()]) for cell
            in self.cells]])
        atom_vol *= 4.0 * np.math.pi / 3.0
        return atom_vol / self.structure.volume()

    def get_neighbor_ordering_parameters(self, shell, weighted):
        """Function to compute the Warren-Cowley short range ordering parameters
        for each atom.
        alpha_{s, i} = 1 - n_{A, s} / (x_A * n_s)
        where n_{A, s} is the number of atoms of type A in shell s, x_A is
        the composition of atom A and n_s is the number of atoms in shell s.
        Optionally, one can weight the contributions of each neighbor based on
        the path weights. See VoronoiCell.get_neighbors_by_walks for further
        discussion.

        Parameters
        ----------
        shell : int
            Index of nearest neighbor shell.
        weighted : bool
            Whether to compute the weighted ordering parameters.

        Returns
        -------
        output : array-like
            Ordering parameters for each atom in cell.

        References
        ----------
        .. [1] J. M. Cowley, "An Approximate Theory of Order in Alloys,
        " Physical Review, vol. 77, no. 5, pp. 669--675, Mar. 1950.
        """
        n_atoms = self.structure.n_atoms()
        n_types = self.structure.n_types()
        output = np.zeros((n_atoms, n_types), dtype=float)

        # Compute the composition.
        x = np.zeros(n_types, dtype=float)
        for atom in self.structure.get_atoms():
            x[atom.get_type()] += 1

        x /= n_atoms

        # Compute the WC parameters for each atom.
        for a in range(n_atoms):
            if weighted:
                # Get the neighbors and weights.
                neighbor_shell = self.cells[a].get_neighbors_by_walks(
                    self.cells, shell)

                # Compute the actual weight of each type.
                n_a = np.zeros(n_types, dtype=float)
                for (k,v) in iteritems(neighbor_shell):
                    n_a[k.get_atom().get_type()] += v

                # Compute the WC parameters.
                for t in range(n_types):
                    if x[t] != 0:
                        output[a][t] = 1 - n_a[t] / x[t]
            else:
                # Get the neighboring atoms.
                neighbors = self.cells[a].get_neighbor_shell(self.cells, shell)

                # Compute the actual number of each type.
                n_a = np.zeros(n_types, dtype=float)
                for neighbor in neighbors:
                    n_a[neighbor.get_atom().get_type()] += 1

                # Compute the WC parameters.
                l_n = len(neighbors)
                for t in range(n_types):
                    if x[t] != 0:
                        output[a][t] = 1 - n_a[t] / l_n / x[t]
        return output

    def warren_cowley_ordering_magnitude(self, shell, weighted):
        """Function to compute the mean deviation in the Warren-Cowley
        parameter for each type for site from 0.
        Consider this as a measure of how "ordered" a structure is. Computed
        as the average of the average the absolute values of the WC
        parameters for each type for each site:
        Sum_{i,j} 1 / [Number of atoms in structure] / [Number of types] * |
        [ WC parameter for type i about atom j] |

        Parameters
        ----------
        shell : int
            Index of neighbor shell (e.g. 1st shell = 1).
        weighted : bool
            Whether to weigh ordering parameters by face area.

        Returns
        -------
        output : float
            Computed value.

        """

        # Get the WC ordering parameters.
        wc = self.get_neighbor_ordering_parameters(shell, weighted)
        return np.sum(np.abs(wc)) / self.structure.n_atoms() / \
               self.structure.n_types()

    def compute_shape_dissimilarity(self, this_shape, reference_shape):
        """Function to compute the similarity of the shape of a cell to a
        reference.

        Computed as the difference between the number of faces with a certain
        number of edges for between a shape and the reference for each type
        of face (defined by the number of edges) divided by the total number
        of faces in the reference shape.

        Example: Reference shape has 12 square faces, this shape has 11
        square faces and two triangular faces. There are three different faces
        (i.e. one missing square face and two extraneous triangular faces).
        Therefore the dissimilarity is: 3 / 12 = 25%.

        Parameters
        ----------
        this_shape : dict
            Shape to be compared.
        reference_shape : dict
            Shape to compare against.

        Returns
        -------
        output : float
            Dissimilarity figure (0 means identical, can be > 1).

        """
        n_diff = 0
        n_faces = 0

        # For all shapes that in reference pattern.
        for (type, count) in iteritems(reference_shape):
            n_faces += count
            if type in this_shape:
                n_diff += abs(count - this_shape[type])
            else:
                n_diff += count

        # For all types in this shape, but not reference pattern.
        types_to_eval = set(this_shape.keys())
        types_to_eval -= set(reference_shape.keys())
        for type in types_to_eval:
            n_diff += this_shape[type]

        return float(n_diff) / n_faces

    def mean_sc_dissimilarity(self):
        """Function to get how dissimilar, on average, coordination polyhedra
        are from super cell.

        Returns
        -------
        output : float
            Average shape dissimilarity from sc.
        """

        # Make reference.
        ref = {4 : 6}

        # Return mean dissimilarity.
        return np.mean([self.compute_shape_dissimilarity(self.cells[
            i].get_polyhedron_shape(), ref) for i in range(
            self.structure.n_atoms())])

    def mean_bcc_dissimilarity(self):
        """Function to get how dissimilar, on average, coordination polyhedra
        are from bcc.

        Returns
        -------
        output : float
            Average shape dissimilarity from bcc.
        """

        # Make reference.
        ref = {6: 8, 4: 6}

        # Return mean dissimilarity.
        return np.mean([self.compute_shape_dissimilarity(self.cells[
            i].get_polyhedron_shape(), ref) for i in range(
            self.structure.n_atoms())])

    def mean_fcc_dissimilarity(self):
        """Function to get how dissimilar, on average, coordination polyhedra
        are from fcc.

        Returns
        -------
        output : float
            Average shape dissimilarity from fcc.

        """

        # Make reference.
        ref = {4: 12}

        # Return mean dissimilarity.
        return np.mean([self.compute_shape_dissimilarity(self.cells[
            i].get_polyhedron_shape(), ref) for i in range(
            self.structure.n_atoms())])

    def neighbor_property_differences(self, property, shell):
        """Function to compute the face-size-weighted difference between
        properties of an atom and its neighbors.
        Computed as:
        sum_i [ Size of face between atom and neighbor i ] * |
        [property of atom] - [property neighbor] | / [ surface area ]
        For neighbors in the 2nd or greater shell,
        VoronoiCell.get_extended_faces is used to find the unique faces
        between the atoms in the N - 1 shell and the Nth shell. So, each Nth
        neighbor atom might have multiple faces and its total "size of face"
        will be defined as the sum of the area of these faces.

        Parameters
        ----------
        property : array-like
            List of property for each atom type.
        shell : int
            Shell to be considered (1 == 1st nearest neighbor shell).

        Returns
        -------
        output : array-like
            Property difference value for each neighbor.

        """

        # Get lookup table of face areas and types.
        neighbors = self.get_neighbor_shell_weights(shell)
        face_types = neighbors[0]
        face_areas = neighbors[1]

        n_atoms = self.structure.n_atoms()
        # Compute average difference for each cell.
        output = np.zeros(n_atoms, dtype=float)
        for a in range(n_atoms):
            # Get the property of this atom.
            my_value = property[self.structure.get_atom(a).get_type()]
            output[a] = np.average([abs(my_value - property[face_types[a][f]])
                        for f in range(len(face_areas[a]))],
                                   weights=face_areas[a])

        return output

    def get_neighbor_shell_weights(self, shell):
        """Function to get the types and weights on neighbors in each shell.
        Uses the path weight argument described in
        VoronoiCell.get_neighbors_by_walks.

        Parameters
        ----------
        shell : int
            Shell being considered. 1 corresponds to the 1st NN,
            2 corresponds to the polyhedron formed by an atom and its 1st shell
            neighbors, etc.

        Returns
        -------
        output : array-like
            Pair of neighbor types and weights.

        """
        n_atoms = self.structure.n_atoms()
        types = np.zeros((n_atoms, ), dtype=np.ndarray)
        weights = np.zeros((n_atoms, ), dtype=np.ndarray)

        for a in range(n_atoms):
            # Get extended faces.
            neighbors = self.cells[a].get_neighbors_by_walks(self.cells, shell)
            l_n = len(neighbors)
            # Store their areas.
            types[a] = np.array([k.get_atom().get_type() for k in
                                 list(neighbors.keys())], dtype=int)
            weights[a] = np.array(list(neighbors.values()), dtype=float)

        return [types, weights]

    def neighbor_property_variances(self, property, shell):
        """Function to get the weighted variance between the properties of each
        atom's neighbors.

        Parameters
        ----------
        property : array-like
            List of property for each type.
        shell : int
            Shell being considered. 1 corresponds to the 1st NN,
            2 corresponds to the polyhedron formed by an atom and its 1st shell
            neighbors, etc.

        Returns
        -------
        output : array-like
            Variance in property of each atom for each neighbor.

        Raises
        ------
        Exception
            If wrong number of property values provided.
        """

        if len(property) != self.structure.n_types():
            raise Exception("Wrong number of property values provided.")

        neighbors = self.get_neighbor_shell_weights(shell)
        types = neighbors[0]
        weights = neighbors[1]

        n_atoms = self.structure.n_atoms()
        # Compute average difference for each cell.
        output = np.zeros(n_atoms, dtype=float)

        for a in range(n_atoms):
            l_t = len(types[a])

            # Get the properties of all neighbors.
            face_props = np.array([property[types[a][f]] for f in range(
                l_t)], dtype=float)

            # Compute the face-weighted mean.
            mean = np.average(face_props, weights=weights[a])

            # Compute the face-weighted variance.
            x = face_props - mean * np.ones(l_t)
            output[a] = np.average(x ** 2, weights=weights[a])
            # print a, mean, output[a]

        return output

    def bond_lengths(self):
        """Function to get the bond lengths for each atom.

        Returns
        -------
        output : array-like
            Bond lengths for each atom in Cartesian units.
        """
        n_atoms = self.structure.n_atoms()
        output = np.array([self.cells[a].get_neighbor_distances() for a in
                           range(n_atoms)])
        return output

    def mean_bond_lengths(self):
        """Function to compute mean bond length for each cell.
        Bond length for each neighbor is weighted by face size.

        Returns
        -------
        output : float
            Mean bond lengths.

        """
        n_atoms = self.structure.n_atoms()
        output = np.array([sum([face.get_area() * face.get_neighbor_distance()
                for face in self.cells[a].get_faces()]) /
                self.cells[a].get_surface_area() for a in range(n_atoms)])
        return output

    def bond_length_variance(self, mean_lengths):
        """Function to compute variance in bond length for each face.
        Computed as the mean absolute deviation of each neighbor distance
        from the mean neighbor distance, weighted by face area.

        Parameters
        ----------
        mean_lengths : float
            Mean bond length for each cell.

        Returns
        -------
        output : array-like
            Bond length variance for each cell.

        """
        n_atoms = self.structure.n_atoms()
        output = np.array([sum([face.get_area() *
                abs(face.get_neighbor_distance() - mean_lengths[a])
                    for face in self.cells[a].get_faces()]) /
                    self.cells[a].get_surface_area() for a in range(n_atoms)])
        return output
