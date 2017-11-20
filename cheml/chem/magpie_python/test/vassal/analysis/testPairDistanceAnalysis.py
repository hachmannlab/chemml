import unittest
from numpy.linalg import norm
from vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell

class testPairDistanceAnalysis(unittest.TestCase):
    def setUp(self):
        self.structure = Cell()
        self.structure.set_basis(lengths=[1, 1, 1], angles=[90, 90, 90])
        self.structure.add_atom(Atom([0, 0, 0], 0))
        self.pda = PairDistanceAnalysis()
        self.pda.analyze_structure(self.structure)

    def tearDown(self):
        self.structure = None
        self.pda = None

    def test_get_all_neighbors_of_atom(self):
        # With orthorhombic basis.
        self.pda.set_cutoff_distance(1.1)
        output = self.pda.get_all_neighbors_of_atom(0)
        self.assertEquals(6, len(output))
        self.assertAlmostEquals(1.0, output[0][1], delta=1e-6)

        # Adding a second atom.
        self.structure.add_atom(Atom([0.5, 0.5, 0.5], 0))
        output = self.pda.get_all_neighbors_of_atom(0)
        self.assertEquals(14, len(output))

        # Altering the basis to something weird.
        new_basis = self.structure.get_basis()
        new_basis[1][0] = 14
        output = self.pda.get_all_neighbors_of_atom(0)
        self.assertEquals(14, len(output))

        # Check that images match up.
        center_pos = self.structure.get_atom(0).get_position_cartesian()
        for image in output:
            v = image[0].get_position() - center_pos
            self.assertAlmostEquals(image[1], norm(v), delta=1e-6)

    def test_PRDF(self):
        # With orthorhombic basis.
        self.pda.set_cutoff_distance(2.1)

        # Run code.
        prdf = self.pda.compute_PRDF(50)
        self.assertEquals(1, len(prdf))
        self.assertEquals(1, len(prdf[0]))
        self.assertEquals(50, len(prdf[0][0]))

        # Make sure that it finds 4 peaks.
        n_peaks = 0
        for val in prdf[0][0]:
            if val > 0:
                n_peaks += 1

        self.assertEquals(4, n_peaks)

        # Add another atom, repeat.
        self.structure.add_atom(Atom([0.5, 0.5, 0.5], 1))

        # Run again.
        prdf = self.pda.compute_PRDF(50)
        self.assertEquals(2, len(prdf))
        self.assertEquals(2, len(prdf[0]))
        self.assertEquals(50, len(prdf[0][0]))

        # Make sure A-B prdf has 2 peaks.
        n_peaks = 0
        for val in prdf[0][1]:
            if val > 0:
                n_peaks += 1

        self.assertEquals(2, n_peaks)

        # Increase basis.
        self.structure.set_basis(lengths=[2, 2, 2], angles=[90, 90, 90])
        self.pda.analyze_structure(self.structure)

        # Run again.
        prdf = self.pda.compute_PRDF(50)
        self.assertEquals(2, len(prdf))
        self.assertEquals(2, len(prdf[0]))
        self.assertEquals(50, len(prdf[0][0]))

        # Make sure A-B prdf has 1 peaks.
        n_peaks = 0
        for val in prdf[0][1]:
            if val > 0:
                n_peaks += 1

        self.assertEquals(1, n_peaks)