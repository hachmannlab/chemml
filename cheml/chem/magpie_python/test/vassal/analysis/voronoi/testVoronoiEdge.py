import unittest
import numpy.testing as np_tst
from vassal.analysis.voronoi.VoronoiEdge import VoronoiEdge
from vassal.analysis.voronoi.VoronoiFace import VoronoiFace
from vassal.data.Atom import Atom
from vassal.data.AtomImage import AtomImage
from vassal.data.Cell import Cell

class testVoronoiEdge(unittest.TestCase):
    def setUp(self):
        # Make a simple crystal.
        self.cell = Cell()
        self.cell.add_atom(Atom([0, 0, 0], 0))

    def tearDown(self):
        self.cell = None

    def test_initialize(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edge.
        edge = VoronoiEdge(face1, face2)

        # Check out properties.
        np_tst.assert_array_almost_equal([-1, 0, 0], edge.get_line(
        ).get_direction())

    def test_is_next(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [1, 0, 0])
        face3 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)

        # Check proper ordering.
        self.assertTrue(edge2.is_ccw(edge2=edge1))
        self.assertFalse(edge1.is_ccw(edge2=edge2))
        self.assertFalse(edge2.is_ccw(edge2=edge2))

    def test_simple_intersection(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [1, 0, 0])
        face3 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [2, 0, 0])
        face4 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [-1, 0, 0])
        face5 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)
        edge3 = VoronoiEdge(face1, face4)
        edge4 = VoronoiEdge(face1, face5)

        # Compute intersection.
        self.assertTrue(VoronoiEdge.compute_intersection(edge1, edge3))
        self.assertTrue(VoronoiEdge.compute_intersection(edge1, edge2))
        self.assertTrue(VoronoiEdge.compute_intersection(edge1, edge4))

        # Test properties.
        np_tst.assert_array_almost_equal([0.5, 0.5, 0.5],
                        edge1.get_start_vertex().get_position())
        np_tst.assert_array_almost_equal([0.5, 0.5, 0.5],
                        edge2.get_end_vertex().get_position())
        self.assertAlmostEquals(1.0, edge1.get_length(), delta=1e-6)

        # Compute intersection that shouldn't happen.
        self.assertFalse(VoronoiEdge.compute_intersection(edge1, edge3))

    def test_join(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [1, 0, 0])
        face3 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [2, 0, 0])
        face4 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)
        edge3 = VoronoiEdge(face1, face4)

        # Join edge1 & edge2.
        VoronoiEdge.compute_intersection(edge1, edge2, just_join=True)
        np_tst.assert_array_almost_equal([0.5, 0.5, 0.5],
                        edge1.get_start_vertex().get_position())
        self.assertEquals(edge2, edge1.get_previous_edge())

        # Join edge1 & edge3.
        VoronoiEdge.compute_intersection(edge1, edge3, just_join=True)
        np_tst.assert_array_almost_equal([1.0, 0.5, 0.5],
                    edge1.get_start_vertex().get_position())
        self.assertEquals(edge3, edge1.get_previous_edge())

    def test_compare(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [1, 0, 0])
        face3 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)

        # Tests.
        self.assertEquals(0, edge1.__cmp__(edge1))
        self.assertTrue(edge1.__cmp__(edge2) != 0)
        self.assertTrue(edge1.__cmp__(edge2) == -1 * edge2.__cmp__(edge1))
        self.assertFalse(edge1.__eq__(edge2))
        self.assertTrue(edge1.__eq__(edge1))

    def test_next_edge(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [-1, 1, 0])
        face3 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [-1, 0, 0])
        face4 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)
        edge3 = VoronoiEdge(face1, face4)

        # Verify that edge2 and edge3 intersect edge1 in the same place.
        p1 = edge1.get_line().intersection(edge2.get_line())
        p2 = edge1.get_line().intersection(edge3.get_line())
        np_tst.assert_array_almost_equal(p1, p2)

        # Verify that when tasked with distinguishing between the two, it
        # recognizes that edge3 is the correct choice
        choices = [edge2, edge3]
        self.assertTrue(edge1.is_ccw(edge2=edge2))
        self.assertEquals(edge3, edge1.find_next_edge(choices))

        # Add more conflicting choices.
        image = AtomImage(self.cell.get_atom(0), [-2, 0, 0])
        face4 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        # A farther edge.
        choices.append(VoronoiEdge(face1, face4))
        image = AtomImage(self.cell.get_atom(0), [1, 0, 0])
        face4 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        # A CW edge.
        choices.append(VoronoiEdge(face1, face4))

        self.assertEquals(edge3, edge1.find_next_edge(choices))

    def test_pair(self):
        # Initialize faces.
        image = AtomImage(self.cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(self.cell.get_atom(0), image, radical=True)
        image = AtomImage(self.cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(self.cell.get_atom(0), image, radical=True)

        # Create edge.
        edge = VoronoiEdge(face1, face2)

        # Create pair.
        pair = edge.generate_pair()

        # Tests.
        self.assertEquals(edge.get_edge_face(), pair.get_intersecting_face())
        self.assertEquals(edge.get_intersecting_face(), pair.get_edge_face())
        dir1 = edge.get_line().get_direction()
        dir2 = -1 * pair.get_line().get_direction()
        np_tst.assert_array_almost_equal(dir1, dir2)