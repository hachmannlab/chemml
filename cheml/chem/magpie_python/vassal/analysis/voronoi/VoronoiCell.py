from collections import OrderedDict
from numpy.linalg import norm
import numpy as np
from vassal.analysis.voronoi.VoronoiEdge import VoronoiEdge
from vassal.analysis.voronoi.VoronoiFace import VoronoiFace
from vassal.data.AtomImage import AtomImage

class VoronoiCell:
    """
    Class describing a single cell in a Voronoi tessellation.
    """
    def __init__(self, atom, radical, faces=None):
        """
        Function to initialize a Voronoi cell for a specific atom. Creates a
        parallelepiped around the atom defined by its interactions with the
        periodic images across the cube face of the cell, or the walls of the
        cell.
        :param atom: Atom at center of cell.
        :param radical: Whether to add faces using the radical plane Voronoi
        tessellation method.
        :param faces: List of faces in this cell.
        """

        # Atom at the center of this cell.
        self.atom = atom

        # List of faces.
        self.faces = faces

        # Whether to use radical plane method.
        self.radical = radical

        # Volume of cell. Cached result.
        self.volume = np.nan

    def get_atom(self):
        """
        Function to get atom at center of this cell.
        :return: Atom.
        """
        return self.atom

    def get_faces(self):
        """
        Function to get all faces.
        :return: List of faces.
        """
        return self.faces

    def n_faces(self):
        """
        Function to get the number of faces.
        :return: Number of faces.
        """
        return len(self.faces)

    def get_vertices(self):
        """
        Function to get all vertices in this cell.
        :return: List of vertices.
        """
        return list(set([face.get_vertices() for face in self.faces]))

    def get_neighbor_types(self):
        """
        Function to get the number of neighbors of each type.
        :return: Ordered dictionary with atom_type, number of neighbors as
        key,value pairs.
        """
        output = OrderedDict()
        for face in self.faces:
            atom_id = face.get_outside_atom().get_atom_id()
            atom_type = self.atom.get_cell().get_atom(atom_id).get_type()
            if atom_type in output:
                output[atom_type] += 1
            else:
                output[atom_type] = 1

        return output

    def get_neighbors(self):
        """
        Function to get list of neighboring atoms. Identifies atoms by both
        their ID and image location.
        :return: List of atom images.
        """
        return np.array([face.get_outside_atom() for face in self.faces])

    def get_neighbor_distances(self):
        """
        Function to get distances between central atom and each neighbor.
        :return: List of distances in cartesian units.
        """
        return np.array([face.get_neighbor_distance() for face in self.faces])

    def get_neighbor_shells(self, cells, index):
        """
        Function to get list of atoms in all coordination cells.
        :param cells: Voronoi cells of all other atoms (arranged by atom ID).
        :param index: Index of largest neighbor shell to be considered.
        :return: All neighbors in that.
        """
        if index < 0:
            # Special case: Bad input.
            raise ValueError("Shell index must be >= 0.")
        elif index == 0:
            # Special case: 0th cell (this atom).
            return [OrderedDict({AtomImage(self.atom, [0, 0, 0]):None})]
        else:
            # General case: i-th shell.
            # Get the previous shell.
            previous_shells = self.get_neighbor_shells(cells, index - 1)

            # For each atom in the previous shell, get its neighbors.
            new_shell = OrderedDict()
            for atom in previous_shells[index - 1]:
                # Get its nearest neighbor according to the diagram.
                neighbors = cells[atom.get_atom_id()].get_neighbors()
                # For each of these neighbors, adjust the image coordinates.
                this_image = atom.get_supercell()
                for neighbor in neighbors:
                    # Get the coordinates relative to the central atom.
                    old_image = neighbor.get_supercell() + this_image

                    # Create a new image, and append it to output.
                    new_image = AtomImage(neighbor.get_atom(), old_image)
                    if new_image not in new_shell:
                        new_shell[new_image] = None

            # Remove all images corresponding to the shell inside this one.
            for shell in previous_shells:
                for image in shell:
                    if image in new_shell:
                        new_shell.pop(image)

            # Append new shell to output.
            previous_shells.append(new_shell)

            return previous_shells

    def get_extended_faces(self, cells, index):
        """
        Function to get faces on the outside of a polyhedron formed by the
        Voronoi cells and its neighbors.
        Note: The coordinates of the center of each face may not be correct.
        :param cells: Voronoi cells of all other atoms (arranged by Atom ID).
        :param index: Index of largest neighbor shell to be included (0 ==
        faces of atom, 1 == faces of atom + 1st nearest neighbor shell, etc.).
        :return: List of faces on the outside.
        """

        if index == 0:
            # Special case: Index == 0.
            return list(self.faces)

        # Get neighbor list.
        neighbor_list = self.get_neighbor_shells(cells, index)

        # Last shell are atoms on the outside.
        outside_images = neighbor_list[index]

        # Get atoms to consider.
        all_images = OrderedDict({[[image for images in neighbor_list for
                                    image in images]]:None})

        # Get the faces of all outside atoms that do not correspond to an
        # atom on the inside of the polyhedron.
        output = []
        for image in outside_images:
            # Get coordinates of this image w.r.t. central atom.
            image_coord = image.get_supercell()
            # Get its Voronoi cell.
            cell = cells[image.get_atom_id()]
            # Iterate over its faces.
            for face in cell.get_faces():
                # Get the image on the outside of this face.
                relative_image = face.get_outside_atom()
                # Get the coordinates of this image relative to the central
                # atom.
                outside_coord = relative_image.get_supercell() + image_coord

                # Get the actual image.
                actual_image = AtomImage(relative_image.get_atom(),
                                         outside_coord)

                # If this image is not inside of the polyhedron, add the face
                #  to the output.
                if actual_image not in all_images:
                    output.append(face)
        return output

    def get_neighbors_by_walks(self, cells, shell):
        """
        Function to get the neighbors in a certain shell, along with weights
        defined by face sizes.
        The neighbor shell of an atom is typically defined as all atoms that
        can be reach ed in no fewer than a certain number of steps along the
        a network defined by faces of a Voronoi tessellation. While this works
        in theory, it is complicated by the fact that Voronoi tessellations
        computed numerically often contain very small faces due to numerical
        problems. These atoms with very small faces would be conventionally
        defined as 2nd nearest neighbors but, due to the small faces,
        would be counted as as 1st nearest neighbors.

        To combat this problem, we weighing each neighbor based on the face
        sizes. For a first neighbor shell, the weight of an atom is associated
        with its the fraction of the surface area of the cell related to the
        area of the face corresponding to it. To extend this idea to shells
        past the 1st, we compute the probability of a walker originating from
        this cell ends up at a certain atom after N, non-backtracking walks
        where the probability that it will take a certain step is related to
        the fraction of the face size area.

        For example, let's assume this cell is part of a simple-cubic crystal,
        where each cell has 6, equally-sized faces. The probability that it
        ends up at any first shell neighbor is 1/6. After this step,
        there are 5 possible steps (no backtracking). So, each walk has an
        equal probability of 1/30. There are six atoms that are two steps in
        the same direction (ex: 2,0,0), which can be reached by only 1 path
        and therefore have 1/30 weight. Additionally, there are 12 atoms that
        can be reached by 2 different sets of two orthogonal steps (ex: 1,1,
        0) that, therefore, have a weight of 2/30.
        :param cells: Voronoi cells of all other atoms (arranged by atom ID).
        :param shell: Which neighbor shell to evaluate.
        :return: List of atoms in that shell, and their path weights. Weights
        will add up to 1.
        """

        # Gather all possible paths.
        # Initialize with first step.
        paths = []
        # Add in starting point.
        paths.append(([AtomImage(self.atom, [0, 0, 0])], 1.0))
        # Increment until desired shell.
        for step in range(shell):
            # Get the new list.
            new_paths = []
            # For each current path.
            for path in paths:
                # Get last atom in current path.
                last_step = path[0][len(path[0]) - 1]
                # Get each possible step.
                new_steps = OrderedDict()
                # Surface area of possible steps.
                surface_area = 0.0
                for face in cells[last_step.get_atom_id()].get_faces():
                    # Adjust image based on position of last atom.
                    new_supercell = face.get_outside_atom().get_supercell() +\
                                    last_step.get_supercell()

                    # Store the next step.
                    next_step = AtomImage(face.get_outside_atom().get_atom(),
                                          new_supercell)
                    area = face.get_area()
                    surface_area += area
                    new_steps[next_step] = area

                # Eliminate backtracking steps.
                for previous_step in path[0]:
                    if previous_step in new_steps:
                        surface_area -= new_steps.pop(previous_step)

                # Create new paths, making sure to update weights.
                for k,v in new_steps.iteritems():
                    # Increment path.
                    new_path = list(path[0])
                    new_path.append(k)

                    # Increment weight.
                    new_weight = path[1] * v / surface_area

                    # Add it to new_paths.
                    new_paths.append((new_path, new_weight))

            # Update paths.
            paths = new_paths

        # Now that all the paths are gathered, output only the last step and
        # weights of all paths that lead to that step.
        output = OrderedDict()
        for path in paths:
            # Get the last step.
            atom = path[0][len(path[0]) - 1]

            # Update map.
            if atom in output:
                output[atom] += path[1]
            else:
                output[atom] = path[1]

        return output

    def get_neighbor_shell(self, cells, index):
        """
        Function to get list of atoms in a certain coordination shell. A
        neighbor shell includes all atoms that are a certain number of
        neighbors away that are not in any smaller neighbor shell (e.g. 2nd
        shell neighbors cannot also be 1st nearest neighbors).
        :param cells: Voronoi cells of all other atoms (arranged by atom ID).
        :param index: Index of neighbor shell.
        :return: All neighbors in that.
        """
        return self.get_neighbor_shells(cells, index)[index]

    def get_num_shared_bonds(self, cell, direction, neighbors):
        """
        Function to compute number of shared bonds. Atoms are defined as
        bonded if they share a face between their Voronoi cells.

        This code determines how many common entries occur between a Voronoi
        cell and a list of neighbor IDs from this cell. Note that in order for
        bonds to be shared, they must connect to the same image of a certain
        atom. That is where the direction parameter comes into play.
        :param cell: Voronoi cell of neighboring atom.
        :param direction: Difference between image of neighboring atom and
        this cell.
        :param neighbors: IDs and Image Positions of all neighboring atoms to
        this cell.
        :return: Number of shared neighbors between this cell and the
        neighboring atom.
        """
        n_shared = 0
        for face in cell.get_faces():
            other_cell_n_id = face.get_outside_atom().get_atom_id()
            other_cell_n_image = face.get_outside_atom().get_supercell() + \
                                 direction
            # Make that image.
            other_cell_n = AtomImage(self.atom.get_cell().get_atom(
                other_cell_n_id), other_cell_n_image)

            # Check against every neighbor of this cell.
            for n in neighbors:
                if other_cell_n.__eq__(n):
                    n_shared += 1

        return n_shared

    def get_coordination_shell_shape(self, cells):
        """
        Function to get the shape of the coordination polyhedron around this
        atom. This is determined by counting the number of "bonds" between
        first neighbor shell atoms. Here, atoms are "bonded" if their Voronoi
        cells share a face. This is similar to get_polyhedron_shape. See
        section H of the paper listed below for more details.

        http://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.134205
        Ward et al. PRB (2013).
        :param cells: Voronoi cells of all other atoms (arranged by atom ID).
        :return: Ordered dictionary containing the number of bonds, number of
        first shell neighbors with that many bonds as key,value pairs.
        """

        # Get IDs of every neighbor.
        neighbors = [face.get_outside_atom() for face in self.faces]

        # Get number of mutual neighbors for each neighbor.
        output = OrderedDict()
        for n in neighbors:
            n_shared = self.get_num_shared_bonds(cells[n.get_atom_id()],
                                                 n.get_supercell(), neighbors)
            if n_shared in output:
                output[n_shared] += 1
            else:
                output[n_shared] = 1

        return output

    def get_polyhedron_shape(self):
        """
        Function to compute the polyhedron shape index. This shape is defined
        by the "Voronoi Index", which is defined as the number of faces with
        a certain number of sides. This is often expressed as
        (n_3, n_4, n_5, n_6), where n_x is the number of faces with x number
        of sides.
        http://rspa.royalsocietypublishing.org/cgi/doi/10.1098/rspa.1970.0190
        Finney (1970).
        :return: Ordered dictionary containing number of edges, number of
        faces with that edge count as key,value pairs.
        """
        output = OrderedDict()
        for face in self.faces:
            n_vertices = face.n_vertices()
            if n_vertices in output:
                output[n_vertices] += 1
            else:
                output[n_vertices] = 1

        return output

    def get_volume(self):
        """
        Function to get volume of this cell
        :return: Volume.
        """
        if np.isnan(self.volume):
            self.volume = 0
            atom_center = self.atom.get_position_cartesian()
            for face in self.faces:
                area = face.get_area()
                from_center = face.get_centroid() - atom_center
                n = face.get_normal()
                n /= norm(n)
                h = np.dot(from_center, n)
                self.volume += area * h / 3.0 # Face normal is away from
                # center.

        return self.volume

    def get_surface_area(self):
        """
        Function to get surface area of cell.
        :return: Surface area.
        """
        return sum([face.get_area() for face in self.faces])

    def get_min_max_vertex_distance(self):
        """
        Function to get the minimum and maximum distance between any two
        vertices.
        :return: Minimum distance, maximum distance.
        """
        vertices = self.get_vertices()
        l_v = len(vertices)
        min_dist = float("inf")
        max_dist = -min_dist
        for i in range(l_v):
            for j in range(i+1, l_v):
                dist = vertices[i].distance_from(vertices[j])
                min_dist = min(dist, min_dist)
                max_dist = max(dist, max_dist)

        return min_dist, max_dist

    def geometry_is_valid(self):
        """
        Function to determine whether the geometry of this structure is sound.
        :return: True if valid, else False.
        """
        for face in self.faces:
            if not face.is_closed():
                return False

            for f in face.get_neighboring_faces():
                if f not in self.faces:
                    return False
        return True

    def compute_cell(self, image_finder, cutoff):
        """
        Function to compute cell, given ability to generate images.
        :param image_finder: Tool to find images within a cutoff.
        :param cutoff: Initial cutoff. Will be increased if too small.
        :return:
        """
        cur_cutoff = cutoff
        n_attempts = 0
        while n_attempts < 4:
            n_attempts += 1
            image_finder.set_cutoff_distance(cur_cutoff)
            # Find all nearby images.
            images = [image[0] for image in
                      image_finder.get_all_neighbors_of_atom(
                          self.atom.get_id())]
            # Compute cell.
            try:
                self.compute_cell_helper(images)
            except Exception:
                cur_cutoff *= 1.5
                continue

            return

        raise Exception("Cell failed to compute.")

    def compute_cell_helper(self, images):
        """
        Function to compute the Voronoi cell, given list of images.
        :param images: List of images to consider, where key is the atomID
        and value is the distance.
        :return:
        """
        # Clear cached volume.
        self.volume = np.nan

        # Get all possible faces.
        possible_faces = self.compute_faces(images)

        # Get the faces corresponding to the direct polyhedron.
        direct_faces = self.compute_direct_neighbors(possible_faces)

        # Construct direct polyhedron.
        for df in direct_faces:
            try:
                df.assemble_face_from_faces(direct_faces)
            except Exception:
                raise Exception("Direct polyhedron failed to construct.")

        self.faces = list(direct_faces)

        # Get the faces that might actually be direct faces.
        possible_indirect_faces = self.compute_possible_indirect_neighbors(
            possible_faces)

        # Use these faces to compute indirect neighbors.
        for face in possible_indirect_faces:
            self.compute_intersection(face)

    def compute_faces(self, images):
        """
        Function to compute the center of the face corresponding to each
        neighbor.
        :param images: List of all images of this atom.
        :return: List of faces.
        """

        # Generate faces.
        output = []
        for image in images:
            # Check if the image is this atom.
            if image.get_atom_id() == self.atom.get_id() and np.array_equal(
                    image.get_supercell(), [0, 0, 0]):
                # If so, skip.
                continue

            # Make the appropriate face.
            try:
                output.append(VoronoiFace(self.atom, image, self.radical))
            except Exception:
                raise RuntimeError("Failure to create face.")

        return output

    def compute_direct_neighbors(self, faces):
        """
        Function to compute list of neighbors for the direct polyhedron and those that could
        contribute to the Voronoi polyhedron.

        See page 85 http://linkinghub.elsevier.com/retrieve/pii
        /0021999178901109
        Brostow, Dessault, Fox. JCP (1978).
        :param faces: List of faces to consider. After this operation,
        any faces that are on the direct polyhedron will be removed and the
        list will be sorted by distance from the center.
        :return: List of faces that are on the direct polyhedron.
        """

        # Sort distance from face to the center.
        faces.sort(cmp=self.compare_faces)

        # The closest face is on the direct polyhedron.
        direct_faces = []
        direct_faces.append(faces.pop(0))

        # Now loop through all faces to find those whose centers are not
        # outside any other face that is on the direct polyhedron.

        to_remove = []
        for face in faces:
            is_inside = True
            for df in direct_faces:
                d = df.position_relative_to_face(face.get_face_center())
                if d >= 0:
                    is_inside = False
                    break

            if is_inside:
                direct_faces.append(face)
                to_remove.append(face)
        for face in to_remove:
            faces.remove(face)
        return direct_faces

    def compare_faces(self, a, b):
        """
        Function to compare two Voronoi faces.
        :param a: Face a.
        :param b: Face b.
        :return: -1 if a < b, +1 if a > b, else 0.
        """
        d1 = a.get_face_distance()
        d2 = b.get_face_distance()

        # Have to check if they are really close. If they are, return 0.
        if (d1 - d2) ** 2 < 1e-30:
            return 0
        elif d1 < d2:
            return -1
        else:
            return +1

    def compute_possible_indirect_neighbors(self, possible_faces):
        """
        Function to get list of faces that might possibly be indirect. You
        must already have computed direct faces and all vertices of the
        direct polyhedron.
        :param possible_faces: List of all possible non-direct faces sorted
        in ascending order. This probably has already been computed by
        compute_direct_neighbors.
        :return: List of faces that might actually be indirect neighbors. Here,
        we find if the face distance is less than the farthest vertex on the
        direct polyhedron.
        """

        # Get list of faces that might be indirect neighbors.
        max_vertex_distance = max([v.get_distance_from_center() for face in
                                   self.faces for v in face.get_vertices()])

        possible_indirect_faces = []
        for face in possible_faces:
            if face.get_face_distance() < max_vertex_distance:
                possible_indirect_faces.append(face)
            else:
                # possible faces is sorted by get_direct_faces.
                break

        return possible_indirect_faces

    def compute_intersection(self, new_face):
        """
        Function to compute intersection between a potential face and this
        cell.
        :param new_face: Potential new face.
        :return: Whether it intersected this cell.
        """

        # Clear cached result.
        self.volume = np.nan

        # Determine whether any faces intersect this new new_face.
        no_intersection = True
        for cur_face in self.faces:
            for v in cur_face.get_vertices():
                d = new_face.position_relative_to_face(v.get_position())
                if d > 0:
                    no_intersection = False
                    break

        if no_intersection:
            return False

        # If it does, store the old face information.
        previous_state = OrderedDict({face: face.get_edges() for face in
                                     self.faces})

        # Attempt to perform intersections.
        try:
            new_edges = []
            to_remove = []
            for c_face in self.faces:
                new_edge = c_face.compute_intersection(new_face)
                if c_face.n_edges() < 3:
                    to_remove.append(c_face)
                else:
                    if new_edge is not None:
                        new_edges.append(new_edge)
            for f in to_remove:
                self.faces.remove(f)
            # Check if we have enough edges.
            if len(new_edges) < 3:
                raise Exception("Not enough edges were formed.")

            # Assemble new face and add it to list of faces.
            new_face.assemble_face_from_edges(new_edges)
            self.faces.append(new_face)

            # Check if geometry is valid.
            if not self.geometry_is_valid():
                raise Exception("Geometry is invalid.")
            return True
        except Exception:
            # Restore previous state.
            self.faces = []
            for face in previous_state:
                face.reset_edges(previous_state[face])
                self.faces.append(face)
            return False

    def remove_face(self, to_remove):
        """
        Function to remove a face from this cell.
        :param to_remove: Face to remove.
        :return:
        """
        if to_remove not in self.faces:
            raise RuntimeError("No such face exists.")

        self.faces.remove(to_remove)
        self.volume = np.nan

        # Find all faces that are currently in contact with this face.
        contacting_faces = to_remove.get_neighboring_faces()
        for face_to_build in contacting_faces:
            cur_edges = face_to_build.get_edges()

            # Compute edges corresponding to the "contacting faces".
            for face in contacting_faces:
                if face.__eq__(face_to_build):
                    continue
                new_edge = VoronoiEdge(face_to_build, face)
                cur_edges.append(new_edge)

            # Remove the edge corresponding to the face being removed.
            for e in cur_edges:
                if e.get_intersecting_face().__eq__(to_remove):
                    cur_edges.remove(e)
            face_to_build.assemble_face_from_edges(cur_edges)