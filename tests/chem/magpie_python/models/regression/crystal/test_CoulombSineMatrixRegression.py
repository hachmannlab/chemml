import numpy as np
import sys
from chemml.chem.magpie_python.models.regression.crystal.CoulombSineMatrixRegression import \
    CoulombSineMatrixRegression
from chemml.chem.magpie_python.vassal.data.Atom import Atom
from chemml.chem.magpie_python.vassal.data.Cell import Cell

class TestCoulombSineMatrixRegression:
    def setup_method(self):
        self.r = CoulombSineMatrixRegression()

    def teardown_method(self):
        self.r = None

    def test_matrix(self):
        # Make a simple structure.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.set_type_name(0, "Al")

        # Compute the sine matrix.
        mat = self.r.compute_coulomb_matrix(structure)
        assert 1 == mat.shape[0]
        assert 1 == mat.shape[1]
        assert abs(0.5 * 13 ** 2.4 - mat[0, 0]) <= 1e-06

        # Add another atom and repeat.
        structure.add_atom(Atom([0.5, 0.5, 0.5], 0))
        mat = self.r.compute_coulomb_matrix(structure)
        assert 2 == mat.shape[0]
        assert 2 == mat.shape[1]

        # Test: Is it insensitive to basis changes.
        new_basis = structure.get_basis()
        new_basis[1, 0] = 12
        structure.set_basis(basis=new_basis)
        assert abs(1.0 - structure.volume()) <= 1e-06
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
        assert 1 == len(mat)

        # Add another atom and repeat.
        structure.add_atom(Atom([0.5, 0.5, 0.5], 0))
        mat = self.r.compute_coulomb_matrix(structure)
        assert 2 == len(mat)

        # Test: Is it insensitive to basis changes.
        new_basis = structure.get_basis()
        new_basis[1, 0] = 12
        structure.set_basis(basis=new_basis)
        assert abs(1.0 - structure.volume()) <= 1e-06
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
        assert abs(1.0 - self.r.compute_similarity(rep1, rep1)) <= 1e-06
        assert abs(1.0 - self.r.compute_similarity(rep2, rep2)) <= 1e-06

        # Check symmetry.
        assert abs(self.r.compute_similarity(rep1, rep2) - self.r.compute_similarity(rep2, rep1)) <= 1e-06

        # Check that similarity between these structures is less than 1.0.
        assert self.r.compute_similarity(rep1, rep2) < 1.0