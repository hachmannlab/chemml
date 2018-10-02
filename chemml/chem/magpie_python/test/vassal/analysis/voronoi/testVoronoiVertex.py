import unittest
from chemml.chem.magpie_python.vassal.analysis.voronoi.VoronoiEdge import VoronoiEdge
from chemml.chem.magpie_python.vassal.analysis.voronoi.VoronoiFace import VoronoiFace
from chemml.chem.magpie_python.vassal.analysis.voronoi.VoronoiVertex import VoronoiVertex
from chemml.chem.magpie_python.vassal.data.Atom import Atom
from chemml.chem.magpie_python.vassal.data.AtomImage import AtomImage
from chemml.chem.magpie_python.vassal.data.Cell import Cell

class testVoronoiVertex(unittest.TestCase):
    def test_creation(self):
        # Make a simple crystal.
        cell = Cell()
        cell.add_atom(Atom([0, 0, 0], 0))

        # Initialize faces.
        image = AtomImage(cell.get_atom(0), [0, 0, 1])
        face1 = VoronoiFace(cell.get_atom(0), image, radical=True)
        image = AtomImage(cell.get_atom(0), [0, 1, 0])
        face2 = VoronoiFace(cell.get_atom(0), image, radical=True)
        image = AtomImage(cell.get_atom(0), [1, 0, 0])
        face3 = VoronoiFace(cell.get_atom(0), image, radical=True)

        # Create edges.
        edge1 = VoronoiEdge(face1, face2)
        edge2 = VoronoiEdge(face1, face3)

        # Create vertex.
        vertex = VoronoiVertex(edge1=edge1, edge2=edge2)

        # Test properties.
        self.assertEqual(edge2, vertex.get_previous_edge())
        self.assertEqual(edge1, vertex.get_next_edge())

        # Make sure the order of edges on creation doesn't matter.
        self.assertEqual(vertex, VoronoiVertex(edge1=edge2, edge2=edge1))

        # Create a new vertex, ensure that it is different.
        edge3 = VoronoiEdge(face2, face3)
        self.assertFalse(vertex.__eq__(VoronoiVertex(edge1=edge3,
                                                     edge2=edge2)))