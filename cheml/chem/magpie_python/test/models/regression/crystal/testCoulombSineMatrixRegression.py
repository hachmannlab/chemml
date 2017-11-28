import unittest
import numpy as np
import sys
from models.regression.crystal.CoulombSineMatrixRegression import \
    CoulombSineMatrixRegression
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell

class testCoulombSineMatrixRegression(unittest.TestCase):
    def setUp(self):
        self.r = CoulombSineMatrixRegression()

    def tearDown(self):
        self.r = None

    def test_matrix(self):
        # Make a simple structure.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.set_type_name(0, "Al")

        # Compute the sine matrix.
        mat = self.r.compute_coulomb_matrix(structure)
        self.assertEquals(1, mat.shape[0])
        self.assertEquals(1, mat.shape[1])
        self.assertAlmostEquals(0.5 * 13 ** 2.4, mat[0, 0],delta=1e-6)

        # Add another atom and repeat.
        structure.add_atom(Atom([0.5, 0.5, 0.5], 0))
        mat = self.r.compute_coulomb_matrix(structure)
        self.assertEquals(2, mat.shape[0])
        self.assertEquals(2, mat.shape[1])

        # Test: Is it insensitive to basis changes.
        new_basis = structure.get_basis()
        new_basis[1, 0] = 12
        structure.set_basis(basis=new_basis)
        self.assertAlmostEquals(1.0, structure.volume(), delta=1e-6)
        mat2 = self.r.compute_coulomb_matrix(structure)
        if np.linalg.norm(mat - mat2) > 1e-6:
            sys.stderr.write("WARNING: Not insensitive to basis changes\n")

    def test_representation(self):
        # Make a simple structure.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.set_type_name(0, "Al")

        # Compute the sine matrix.
        mat = self.r.compute_representation(structure)
        self.assertEquals(1, len(mat))

        # Add another atom and repeat.
        structure.add_atom(Atom([0.5, 0.5, 0.5], 0))
        mat = self.r.compute_coulomb_matrix(structure)
        self.assertEquals(2, len(mat))

        # Test: Is it insensitive to basis changes.
        new_basis = structure.get_basis()
        new_basis[1, 0] = 12
        structure.set_basis(basis=new_basis)
        self.assertAlmostEquals(1.0, structure.volume(), delta=1e-6)
        mat2 = self.r.compute_representation(structure)
        if np.sum(mat - mat2) / len(mat) > 1e-6:
            sys.stderr.write("WARNING: Not insensitive to basis changes\n")

    def test_distance(self):
        # Make two simple structures.
        structure1 = Cell()
        structure1.add_atom(Atom([0, 0, 0], 0))
        structure1.set_type_name(0, "Al")

        structure2 = Cell()
        structure2.add_atom(Atom([0, 0, 0], 0))
        structure2.set_type_name(0, "Ni")

        # Compute representations of each structure.
        rep1 = self.r.compute_representation(structure1)
        rep2 = self.r.compute_representation(structure2)

        # Check that similarity between identical structures is 1.0.
        self.assertAlmostEquals(1.0, self.r.compute_similarity(rep1, rep1),
                                delta=1e-6)
        self.assertAlmostEquals(1.0, self.r.compute_similarity(rep2, rep2),
                                delta=1e-6)

        # Check symmetry.
        self.assertAlmostEquals(self.r.compute_similarity(rep1, rep2),
                                self.r.compute_similarity(rep2, rep1),
                                delta=1e-6)

        # Check that similarity between these structures is less than 1.0.
        self.assertTrue(self.r.compute_similarity(rep1, rep2) < 1.0)