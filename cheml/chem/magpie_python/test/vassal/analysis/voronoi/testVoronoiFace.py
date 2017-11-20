import unittest
import numpy.testing as np_tst
from vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis
from vassal.analysis.voronoi.VoronoiFace import VoronoiFace
from vassal.data.Atom import Atom
from vassal.data.AtomImage import AtomImage
from vassal.data.Cell import Cell
import math

class testVoronoiFace(unittest.TestCase):
    def setUp(self):
        # Make a simple crystal.
        self.cell = Cell()
        self.cell.add_atom(Atom([0, 0, 0], 0))

    def tearDown(self):
        self.cell = None

    def test_initialize(self):
        # Initialize face.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Check out properties.
        np_tst.assert_array_almost_equal([0, 0, 0.5], face.get_face_center())
        self.assertAlmostEquals(0.5, face.get_face_distance(), delta=1e-6)
        np_tst.assert_array_almost_equal([0, 0, 1], face.get_normal())

    def test_simple_assemble(self):
        # Create face to be assembled.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face_to_assemble = VoronoiFace(self.cell.get_atom(0), image,
                                      radical=True)

        # Initialize faces.
        faces = []
        faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [-1, 0, 0]), radical=True))
        faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [1, 0, 0]), radical=True))
        faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [0, -1, 0]), radical=True))
        faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [0, 1, 0]), radical=True))


        # Attempt to assemble face.
        self.assertTrue(face_to_assemble.assemble_face_from_faces(faces))

        # Check out properties.
        self.assertEquals(4, face_to_assemble.n_edges())
        self.assertAlmostEquals(1.0, face_to_assemble.get_area(), delta=1e-6)

    def test_FCC_simple_assemble(self):
        # Make FCC crystal.
        self.cell.add_atom(Atom([0.5, 0.5, 0.0], 0))
        self.cell.add_atom(Atom([0.5, 0.0, 0.5], 0))
        self.cell.add_atom(Atom([0.0, 0.5, 0.5], 0))

        # Create face to be assembled.
        image = AtomImage(self.cell.get_atom(1), [0, 0, 0])
        face_to_assemble = VoronoiFace(self.cell.get_atom(0), image,
                                       radical=True)

        # Initialize faces that will be its neighbors.
        neighbor_faces = []
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [0, 0, -1]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, 0, -1]), radical=True))

        # Attempt to assemble face.
        self.assertTrue(face_to_assemble.assemble_face_from_faces(
            neighbor_faces))

        # Checkout properties.
        self.assertEquals(4, face_to_assemble.n_edges())

        # Now attempt to assemble face considering all possible neighbors.
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [0, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [-1, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [-1, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [-1, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [-1, 0, -1]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, -1, -1]), radical=True))

        # Attempt to assemble face.
        self.assertTrue(face_to_assemble.assemble_face_from_faces(
            neighbor_faces))

        # Checkout properties.
        self.assertEquals(4, face_to_assemble.n_edges())

    def test_FCC_direct_assemble(self):
        # Make FCC crystal.
        self.cell.add_atom(Atom([0.5, 0.5, 0.0], 0))
        self.cell.add_atom(Atom([0.5, 0.0, 0.5], 0))
        self.cell.add_atom(Atom([0.0, 0.5, 0.5], 0))

        # Create face to be assembled.
        image = AtomImage(self.cell.get_atom(1), [0, 0, 0])
        face_to_assemble = VoronoiFace(self.cell.get_atom(0), image,
                                       radical=True)

        # Initialize faces that are direct neighbors of atom 0.
        neighbor_faces = []
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [0, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [-1, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(1), [-1, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [-1, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [-1, 0, -1]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(2), [0, 0, -1]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, 0, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, 0, -1]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, -1, 0]), radical=True))
        neighbor_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(3), [0, -1, -1]), radical=True))

        # Attempt to assemble face.
        self.assertTrue(face_to_assemble.assemble_face_from_faces(
            neighbor_faces))

        # Checkout properties.
        self.assertEquals(4, face_to_assemble.n_edges())

        # Checkout properties.
        self.assertEquals(4, face_to_assemble.n_edges())

        # Now, see if all of the faces assemble.
        volume = 0.0
        for i, face in enumerate(neighbor_faces):
            self.assertTrue(face.assemble_face_from_faces(neighbor_faces))
            self.assertEquals(4, face.n_edges())
            # print i, face.get_outside_atom(), face.get_area(), \
            #     face.get_face_distance()
            area = face.get_area()
            volume += face.get_area() * face.get_face_distance() / 3.0

        self.assertAlmostEquals(0.25, volume, delta=1e-6)

    def test_FCC_full_assemble(self):
        # Make FCC crystal.
        self.cell.add_atom(Atom([0.5, 0.5, 0.0], 0))
        self.cell.add_atom(Atom([0.5, 0.0, 0.5], 0))
        self.cell.add_atom(Atom([0.0, 0.5, 0.5], 0))

        # Create face to be assembled.
        image = AtomImage(self.cell.get_atom(1), [0, 0, 0])
        face_to_assemble = VoronoiFace(self.cell.get_atom(0), image,
                                       radical=True)

        # Get neighbor finding tool.
        imageFinder = PairDistanceAnalysis()
        imageFinder.set_cutoff_distance(3.0)
        imageFinder.analyze_structure(self.cell)

        # Initialize any face.
        neighbor_faces = []
        for new_image in imageFinder.get_all_neighbors_of_atom(0):
            face = VoronoiFace(self.cell.get_atom(0), new_image[0],
                               radical=True)
            neighbor_faces.append(face)

        # Attempt to assemble face.
        self.assertTrue(face_to_assemble.assemble_face_from_faces(
            neighbor_faces))

        # Checkout properties.
        self.assertEquals(4, face_to_assemble.n_edges())

    def test_intersections(self):
        # Create test face.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 2])
        test_face = VoronoiFace(self.cell.get_atom(0), image,
                                       radical=True)

        # Initialize faces.
        original_faces = []
        original_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [-1, 0, 0]), radical=True))
        original_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [2, 0, 0]), radical=True))
        original_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [0, -1, 0]), radical=True))
        original_faces.append(VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [0, 1, 0]), radical=True))
        test_face.assemble_face_from_faces(original_faces)

        self.assertAlmostEquals(1.5, test_face.get_area(), delta=1e-6)

        # Ensure that a non-intersecting face returns "None".
        self.assertEquals(None, test_face.compute_intersection(VoronoiFace(
            self.cell.get_atom(0), AtomImage(self.cell.get_atom(0), [3, 0,
                                                        0]), radical=True)))

        # Check intersection with a face that cuts off an edge completely.
        new_face = VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [1, 0, 0]), radical=True)
        self.assertAlmostEquals(1.0, test_face.get_cut_length(new_face),
                                delta=1e-6)

        new_edge = test_face.compute_intersection(new_face)
        self.assertTrue(new_edge is not None)
        self.assertEquals(new_face, new_edge.get_edge_face())
        self.assertEquals(test_face, new_edge.get_intersecting_face())
        self.assertTrue(test_face.is_closed())
        self.assertAlmostEquals(1.0, test_face.get_area(), delta=1e-6)

        # Compute intersection with a face that adds a new edge.
        test_face.assemble_face_from_faces(original_faces)
        new_face = VoronoiFace(self.cell.get_atom(0), AtomImage(
            self.cell.get_atom(0), [1, 1, 0]), radical=True)
        self.assertAlmostEquals(math.sqrt(2)/2, test_face.get_cut_length(
            new_face), delta=1e-6)

        new_edge = test_face.compute_intersection(new_face)
        self.assertTrue(new_edge is not None)
        self.assertEquals(new_face, new_edge.get_edge_face())
        self.assertEquals(test_face, new_edge.get_intersecting_face())
        self.assertEquals(5, test_face.n_edges())