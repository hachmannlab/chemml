import unittest
from chemml.chem.magpie_python.models.regression.crystal.PRDFRegression import PRDFRegression
from chemml.chem.magpie_python.vassal.data.Atom import Atom
from chemml.chem.magpie_python.vassal.data.Cell import Cell

class testPRDFRegression(unittest.TestCase):
    def test_distance(self):
        # Make two simple structures.
        structure1 = Cell()
        structure1.add_atom(Atom([0, 0, 0], 0))
        structure1.set_type_name(0, "Al")

        structure2 = Cell()
        structure2.add_atom(Atom([0, 0, 0], 0))
        structure2.set_type_name(0, "Ni")

        r = PRDFRegression()

        # Compute representations of each structure.
        rep1 = r.compute_representation(structure1)
        rep2 = r.compute_representation(structure2)

        # Check that similarity between identical structures is 1.0.
        self.assertAlmostEqual(1.0, r.compute_similarity(rep1, rep1),
                                delta=1e-6)
        self.assertAlmostEqual(1.0, r.compute_similarity(rep2, rep2),
                                delta=1e-6)

        # Check symmetry.
        self.assertAlmostEqual(r.compute_similarity(rep1, rep2),
                                r.compute_similarity(rep2, rep1),
                                delta=1e-6)

        # Check that similarity between these structures is less than 1.0.
        self.assertTrue(r.compute_similarity(rep1, rep2) < 1.0)

        # Check that the similarity is, in fact, 0
        self.assertAlmostEqual(0, r.compute_similarity(rep1, rep2),
                                delta=1e-6)