import unittest
from .....vassal.analysis.voronoi.VoronoiCell import VoronoiCell
from .....vassal.analysis.voronoi.VoronoiFace import VoronoiFace
from .....vassal.data.Atom import Atom
from .....vassal.data.AtomImage import AtomImage
from .....vassal.data.Cell import Cell
from .....vassal.geometry.Plane import Plane
from .....vassal.util.VectorCombinationComputer import \
    VectorCombinationComputer

class testVoronoiCell(unittest.TestCase):
    def test_supercell(self):
        # Create cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))

        # Create cell for atom1.
        cell = VoronoiCell(structure.get_atom(0), radical=True)

        # Compute faces.
        images = [AtomImage(structure.get_atom(0), sc) for sc in
                  VectorCombinationComputer(
                    structure.get_lattice_vectors(),
                      1.1).get_supercell_coordinates()]
        faces = cell.compute_faces(images)

        # Get direct neighbors.
        direct_faces = cell.compute_direct_neighbors(faces)

        # Simple tests.
        self.assertEquals(6, len(direct_faces))
        self.assertEquals(len(images) - 6 - 1, len(faces))

        # Make sure direct faces match up with expectations.
        neighboring_faces = []
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [1, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [-1, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [0, 1, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [0, -1, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [0, 0, 1]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(0),
                                                       [0, 0, -1]),
                                             radical=True))

        # Test whether they are all there.
        for f in neighboring_faces:
            direct_faces.remove(f)
        self.assertTrue(len(direct_faces) == 0)

    def test_FCC(self):
        # Create cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0], 0))
        structure.add_atom(Atom([0.5, 0, 0.5], 0))
        structure.add_atom(Atom([0, 0.5, 0.5], 0))

        # Create cell for atom1.
        cell = VoronoiCell(structure.get_atom(0), radical=True)

        # Compute faces.
        images = [AtomImage(structure.get_atom(i), sc) for i in range(4) for
                  sc in VectorCombinationComputer(
                      structure.get_lattice_vectors(),
                      4.0).get_supercell_coordinates()]
        faces = cell.compute_faces(images)

        # Get direct neighbors.
        direct_faces = cell.compute_direct_neighbors(faces)
        self.assertTrue(direct_faces[0].get_face_distance() <= direct_faces[
            -1].get_face_distance())

        # Simple tests.
        self.assertEquals(12, len(direct_faces))
        self.assertEquals(len(images) - 12 - 1, len(faces))

        # Make sure direct faces match up with expectations.
        neighboring_faces = []
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(1),
                                                       [0, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(1),
                                                       [0, -1, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(1),
                                                       [-1, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(1),
                                                       [-1, -1, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(2),
                                                       [-1, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(2),
                                                       [-1, 0, -1]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(2),
                                                       [0, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(2),
                                                       [0, 0, -1]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(3),
                                                       [0, 0, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(3),
                                                       [0, 0, -1]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(3),
                                                       [0, -1, 0]),
                                             radical=True))
        neighboring_faces.append(VoronoiFace(structure.get_atom(0),
                                             AtomImage(structure.get_atom(3),
                                                       [0, -1, -1]),
                                             radical=True))
        # Test whether they are all there.
        for f in neighboring_faces:
            direct_faces.remove(f)
        self.assertTrue(len(direct_faces) == 0)

    def test_intersection(self):
        # Create cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))

        # Create cell for atom1.
        cell = VoronoiCell(structure.get_atom(0), radical=True)

        # Make sure direct faces match up with expectations.
        neighboring_faces = []
        neighboring_faces.append(AtomImage(structure.get_atom(0), [1, 0, 0]))
        neighboring_faces.append(AtomImage(structure.get_atom(0), [-1, 0, 0]))
        neighboring_faces.append(AtomImage(structure.get_atom(0), [0, 1, 0]))
        neighboring_faces.append(AtomImage(structure.get_atom(0), [0, -1, 0]))
        neighboring_faces.append(AtomImage(structure.get_atom(0), [0, 0, 2]))
        neighboring_faces.append(AtomImage(structure.get_atom(0), [0, 0, -1]))

        # Assemble cell.
        cell.compute_cell_helper(neighboring_faces)

        # Perform cut.
        cut_face = VoronoiFace(cell.get_atom(), AtomImage(cell.get_atom(),
                    [0, 0, 1]), radical=True)
        cell.compute_intersection(cut_face)

        # Check results.
        self.assertTrue(cut_face.is_closed())
        self.assertEquals(4, cut_face.n_edges())
        self.assertTrue(cell.geometry_is_valid())
        self.assertEquals(6, cell.n_faces())
        self.assertAlmostEquals(1.0, cell.get_volume(), delta=1e-6)

    def test_edge_replacement(self):
        # Create cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))

        # Create cell for atom1.
        cell = VoronoiCell(structure.get_atom(0), radical=True)

        # Compute faces.
        images = [AtomImage(structure.get_atom(0), sc) for sc in
                      VectorCombinationComputer(
                          structure.get_lattice_vectors(),
                          1.1).get_supercell_coordinates()]
        cell.compute_cell_helper(images)

        # Make sure it turned out OK.
        self.assertAlmostEquals(1.0, cell.get_volume(), delta=1e-6)

        # Find position of atom that will take corner off.
        p = Plane(p1=(0.4, 0.5, 0.5), p2=(0.5, 0.4, 0.5), p3=(0.5, 0.5, 0.4),
                  tolerance=1e-6)
        atm_pos = p.project([0, 0, 0])
        atm_pos *= 2
        structure.add_atom(Atom(atm_pos, 0))

        # Cut off the corner.
        cell.compute_intersection(VoronoiFace(cell.get_atom(), AtomImage(
            structure.get_atom(1), [0, 0, 0]), radical=False))
        vol = cell.get_volume()
        self.assertEquals(7, cell.n_faces())

        # Compute a cell that will cut off just slightly more area.
        p = Plane(p1=(0.4, 0.5, 0.5), p2=(0.5, 0.4, 0.5), p3=(0.5, 0.5, 0.3),
                  tolerance=1e-6)
        atm_pos = p.project([0, 0, 0])
        atm_pos *= 2
        structure.add_atom(Atom(atm_pos, 0))
        cell.compute_intersection(VoronoiFace(cell.get_atom(), AtomImage(
            structure.get_atom(2), [0, 0, 0]), radical=False))
        self.assertEquals(7, cell.n_faces())
        self.assertTrue(cell.geometry_is_valid())
        self.assertTrue(cell.get_volume() < vol)

    def test_vertex_replacement(self):
        # Create cell.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))

        # Create cell for atom1.
        cell = VoronoiCell(structure.get_atom(0), radical=True)

        # Compute faces.
        images = [AtomImage(structure.get_atom(0), sc) for sc in
                  VectorCombinationComputer(
                      structure.get_lattice_vectors(),
                      1.1).get_supercell_coordinates()]
        cell.compute_cell_helper(images)

        # Make sure it turned out OK.
        self.assertAlmostEquals(1.0, cell.get_volume(), delta=1e-6)

        # Find position of atom that will take corner off.
        p = Plane(p1=(0.4, 0.5, 0.5), p2=(0.5, 0.4, 0.5), p3=(0.5, 0.5, 0.4),
                  tolerance=1e-6)
        atm_pos = p.project([0, 0, 0])
        atm_pos *= 2
        structure.add_atom(Atom(atm_pos, 0))

        # Cut off the corner.
        cell.compute_intersection(VoronoiFace(cell.get_atom(), AtomImage(
            structure.get_atom(1), [0, 0, 0]), radical=False))
        vol = cell.get_volume()
        self.assertEquals(7, cell.n_faces())

        # Compute a cell that will cut off just slightly more area.
        p = Plane(p1=(0.4, 0.5, 0.5), p2=(0.5, 0.35, 0.5), p3=(0.5, 0.5, 0.35),
                  tolerance=1e-6)
        atm_pos = p.project([0, 0, 0])
        atm_pos *= 2
        structure.add_atom(Atom(atm_pos, 0))
        new_face = VoronoiFace(cell.get_atom(), AtomImage(
            structure.get_atom(2), [0, 0, 0]), radical=False)
        self.assertTrue(cell.compute_intersection(new_face))
        self.assertEquals(7, cell.n_faces())
        self.assertTrue(cell.get_volume() < vol)
        self.assertTrue(cell.geometry_is_valid())

        # Remove that face.
        cell.remove_face(new_face)
        self.assertEquals(6, cell.n_faces())
        self.assertEquals(6, cell.get_polyhedron_shape()[4])
        self.assertAlmostEquals(1.0, cell.get_volume(), delta=1e-6)
        self.assertTrue(cell.geometry_is_valid())