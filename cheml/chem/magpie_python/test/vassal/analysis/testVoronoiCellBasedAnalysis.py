import unittest
import numpy as np
import math
from ....vassal.analysis.VoronoiCellBasedAnalysis import \
    VoronoiCellBasedAnalysis
from ....vassal.data.Atom import Atom
from ....vassal.data.Cell import Cell
import numpy.testing as np_tst

class testVoronoiCellBasedAnalysis(unittest.TestCase):
    def test_BCC(self):
        # Structure of bcc.
        structure = Cell()
        atom = Atom([0, 0, 0], 0)
        structure.add_atom(atom)
        atom = Atom([0.5, 0.5, 0.5], 0)
        structure.add_atom(atom)

        # Prepare.
        tool = VoronoiCellBasedAnalysis(radical=True)
        tool.analyze_structure(structure)

        # Check results.
        n_eff = 11.95692194
        np_tst.assert_array_almost_equal([n_eff, n_eff],
                            tool.get_effective_coordination_numbers())
        self.assertAlmostEquals(14.0, tool.face_count_average(), delta=1e-2)
        self.assertAlmostEquals(0.0, tool.face_count_variance(), delta=1e-2)
        self.assertAlmostEquals(14.0, tool.face_count_minimum(), delta=1e-2)
        self.assertAlmostEquals(14.0, tool.face_count_maximum(), delta=1e-2)
        self.assertAlmostEquals(1, len(tool.get_unique_polyhedron_shapes()),
                                delta=1e-2)
        self.assertAlmostEquals(0.0, tool.volume_variance(), delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_minimum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_maximum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.68, tool.max_packing_efficiency(),
                                delta=1e-2)
        self.assertAlmostEquals(0, tool.mean_bcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(14.0 / 12.0, tool.mean_fcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(8.0 / 6.0, tool.mean_sc_dissimilarity(),
                                delta=1e-2)
        bond_lengths = tool.bond_lengths()
        self.assertEquals(2, len(bond_lengths))
        self.assertEquals(14, len(bond_lengths[0]))
        self.assertAlmostEquals(math.sqrt(3) / 2 , bond_lengths[0][0],
                                delta=1e-6)
        self.assertAlmostEquals(1.0, bond_lengths[0][12], delta=1e-6)
        mean_bond_lengths = tool.mean_bond_lengths()
        var_bond_lengths = tool.bond_length_variance(mean_bond_lengths)
        self.assertTrue(var_bond_lengths[0] > 0)

    def test_B2(self):
        # Structure of bcc.
        structure = Cell()
        atom = Atom([0, 0, 0], 0)
        structure.add_atom(atom)
        atom = Atom([0.5, 0.5, 0.5], 1)
        structure.add_atom(atom)

        # Prepare.
        tool = VoronoiCellBasedAnalysis(radical=True)
        tool.analyze_structure(structure)

        # Check results.
        n_eff = 11.95692194
        np_tst.assert_array_almost_equal([n_eff, n_eff],
                            tool.get_effective_coordination_numbers())
        self.assertAlmostEquals(14.0, tool.face_count_average(), delta=1e-2)
        self.assertAlmostEquals(0.0, tool.face_count_variance(), delta=1e-2)
        self.assertAlmostEquals(14.0, tool.face_count_minimum(), delta=1e-2)
        self.assertAlmostEquals(14.0, tool.face_count_maximum(), delta=1e-2)
        self.assertAlmostEquals(1, len(tool.get_unique_polyhedron_shapes()),
                                delta=1e-2)
        self.assertAlmostEquals(0.0, tool.volume_variance(), delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_minimum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_maximum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.68, tool.max_packing_efficiency(),
                                delta=1e-2)
        self.assertAlmostEquals(0, tool.mean_bcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(14.0 / 12.0, tool.mean_fcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(8.0 / 6.0, tool.mean_sc_dissimilarity(),
                                delta=1e-2)
        bond_lengths = tool.bond_lengths()
        self.assertEquals(2, len(bond_lengths))
        self.assertEquals(14, len(bond_lengths[0]))
        self.assertAlmostEquals(math.sqrt(3) / 2 , bond_lengths[0][0],
                                delta=1e-6)
        self.assertAlmostEquals(1.0, bond_lengths[0][12], delta=1e-6)
        mean_bond_lengths = tool.mean_bond_lengths()
        var_bond_lengths = tool.bond_length_variance(mean_bond_lengths)
        self.assertTrue(var_bond_lengths[0] > 0)

        # Check ordering parameters (against values computed by hand).
        np_tst.assert_array_almost_equal([0.142857, -0.142857],
                                         tool.get_neighbor_ordering_parameters(
                                             1, False)[0], decimal=2)
        np_tst.assert_array_almost_equal([-0.04, 0.04],
                                         tool.get_neighbor_ordering_parameters(
                                             2, False)[0], decimal=2)
        np_tst.assert_array_almost_equal([0.551982, -0.551982],
                                         tool.get_neighbor_ordering_parameters(
                                             1, True)[0], decimal=2)
        np_tst.assert_array_almost_equal([-0.253856, 0.253856],
                                         tool.get_neighbor_ordering_parameters(
                                             2, True)[0], decimal=2)
        self.assertAlmostEquals(0.142857,
                                tool.warren_cowley_ordering_magnitude(1,
                                                                      False),
                                delta=1e-2)
        self.assertAlmostEquals(0.04,
                                tool.warren_cowley_ordering_magnitude(2,
                                                                      False),
                                delta=1e-2)
        self.assertAlmostEquals(0.551982,
                                tool.warren_cowley_ordering_magnitude(1,
                                                                      True),
                                delta=1e-2)
        self.assertAlmostEquals(0.253856,
                                tool.warren_cowley_ordering_magnitude(2,
                                                                      True),
                                delta=1e-2)

    def test_B1(self):
        # Structure of rocksalt.
        structure = Cell()
        basis = np.zeros((3, 3))
        basis[0] = np.array([0, 0.5, 0.5])
        basis[1] = np.array([0.5, 0, 0.5])
        basis[2] = np.array([0.5, 0.5, 0])
        structure.set_basis(basis=basis)
        atom = Atom([0, 0, 0], 0)
        structure.add_atom(atom)
        atom = Atom([0.5, 0.5, 0.5], 1)
        structure.add_atom(atom)

        # Prepare.
        tool = VoronoiCellBasedAnalysis(radical=True)
        tool.analyze_structure(structure)

        # Check results.
        n_eff = 6
        np_tst.assert_array_almost_equal([n_eff, n_eff],
                            tool.get_effective_coordination_numbers())
        self.assertAlmostEquals(6.0, tool.face_count_average(), delta=1e-2)
        self.assertAlmostEquals(0.0, tool.face_count_variance(), delta=1e-2)
        self.assertAlmostEquals(6.0, tool.face_count_minimum(), delta=1e-2)
        self.assertAlmostEquals(6.0, tool.face_count_maximum(), delta=1e-2)
        self.assertAlmostEquals(1, len(tool.get_unique_polyhedron_shapes()),
                                delta=1e-2)
        self.assertAlmostEquals(0.0, tool.volume_variance(), delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_minimum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.5, tool.volume_fraction_maximum(),
                                delta=1e-2)
        np_tst.assert_array_almost_equal([1, -1],
                                         tool.get_neighbor_ordering_parameters(
                                             1, False)[0], decimal=2)
        np_tst.assert_array_almost_equal([-1, 1],
                                         tool.get_neighbor_ordering_parameters(
                                             2, False)[0], decimal=2)
        np_tst.assert_array_almost_equal([1, -1],
                                         tool.get_neighbor_ordering_parameters(
                                             3, False)[0], decimal=2)
        np_tst.assert_array_almost_equal([-1, 1],
                                         tool.get_neighbor_ordering_parameters(
                                             2, True)[0], decimal=2)
        np_tst.assert_array_almost_equal([1, -1],
                                         tool.get_neighbor_ordering_parameters(
                                             3, True)[0], decimal=2)
        np_tst.assert_array_almost_equal([1, -1],
                                         tool.get_neighbor_ordering_parameters(
                                             3, True)[0], decimal=2)
        self.assertAlmostEquals(1,
                                tool.warren_cowley_ordering_magnitude(1,
                                                                      False),
                                delta=1e-2)
        self.assertAlmostEquals(1,
                                tool.warren_cowley_ordering_magnitude(2,
                                                                      False),
                                delta=1e-2)
        self.assertAlmostEquals(1,
                                tool.warren_cowley_ordering_magnitude(1,
                                                                      True),
                                delta=1e-2)
        self.assertAlmostEquals(1,
                                tool.warren_cowley_ordering_magnitude(2,
                                                                      True),
                                delta=1e-2)
        bond_lengths = tool.bond_lengths()
        self.assertEquals(2, len(bond_lengths))
        self.assertEquals(6, len(bond_lengths[0]))
        self.assertAlmostEquals(0.5 , bond_lengths[0][0], delta=1e-6)
        mean_bond_lengths = tool.mean_bond_lengths()
        self.assertEquals(2, len(mean_bond_lengths))
        self.assertAlmostEquals(0.5, mean_bond_lengths[0], delta=1e-6)
        var_bond_lengths = tool.bond_length_variance(mean_bond_lengths)
        self.assertAlmostEquals(0, var_bond_lengths[0], delta=1e-6)

        # Neighbor property attributes.
        np_tst.assert_array_almost_equal([1, 1],
                                         tool.neighbor_property_differences([
                                             0, 1], 1))
        np_tst.assert_array_almost_equal([0, 0],
                                         tool.neighbor_property_differences([
                                             0, 1], 2))
        np_tst.assert_array_almost_equal([0, 0],
                                         tool.neighbor_property_variances([
                                             0, 1], 1))
        np_tst.assert_array_almost_equal([0, 0],
                                         tool.neighbor_property_variances([
                                             0, 1], 2))

    def test_L12(self):
        # Structure of L12.
        structure = Cell()
        atom = Atom([0, 0, 0], 1)
        structure.add_atom(atom)
        atom = Atom([0.5, 0.5, 0], 0)
        structure.add_atom(atom)
        atom = Atom([0.5, 0, 0.5], 0)
        structure.add_atom(atom)
        atom = Atom([0, 0.5, 0.5], 0)
        structure.add_atom(atom)

        # Prepare.
        tool = VoronoiCellBasedAnalysis(radical=True)
        tool.analyze_structure(structure)

        # Check results.
        np_tst.assert_array_almost_equal([12, 12, 12, 12],
                            tool.get_effective_coordination_numbers())
        self.assertAlmostEquals(12.0, tool.face_count_average(), delta=1e-2)
        self.assertAlmostEquals(0.0, tool.face_count_variance(), delta=1e-2)
        self.assertAlmostEquals(12.0, tool.face_count_minimum(), delta=1e-2)
        self.assertAlmostEquals(12.0, tool.face_count_maximum(), delta=1e-2)
        self.assertAlmostEquals(1, len(tool.get_unique_polyhedron_shapes()),
                                delta=1e-2)
        self.assertAlmostEquals(0.0, tool.volume_variance(), delta=1e-2)
        self.assertAlmostEquals(0.25, tool.volume_fraction_minimum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.25, tool.volume_fraction_maximum(),
                                delta=1e-2)
        self.assertAlmostEquals(0.74, tool.max_packing_efficiency(),
                                delta=1e-2)
        self.assertAlmostEquals(1, tool.mean_bcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(0, tool.mean_fcc_dissimilarity(),
                                delta=1e-2)
        self.assertAlmostEquals(1, tool.mean_sc_dissimilarity(),
                                delta=1e-2)
        bond_lengths = tool.bond_lengths()
        self.assertEquals(4, len(bond_lengths))
        self.assertEquals(12, len(bond_lengths[0]))
        self.assertAlmostEquals(math.sqrt(2) / 2, bond_lengths[0][0],
                                delta=1e-6)
        mean_bond_lengths = tool.mean_bond_lengths()
        var_bond_lengths = tool.bond_length_variance(mean_bond_lengths)
        self.assertAlmostEquals(0, var_bond_lengths[0], delta=1e-6)

        # Neighbor analysis results.
        neigh_diff = tool.neighbor_property_differences([-1, 1], 1)
        np_tst.assert_array_almost_equal([2, 8.0 / 12.0, 2.0 / 3, 2.0 / 3],
                                         neigh_diff, decimal=5)
        neigh_diff = tool.neighbor_property_differences([-1, 1], 2)
        np_tst.assert_array_almost_equal([192.0 / 132, 64.0 / 132.0, 64.0 /
                                          132, 64.0/ 132], neigh_diff,
                                         decimal=5)

        neigh_var = tool.neighbor_property_variances([-1, 1], 1)
        # Type 0 has 8 type 0 NNs and 4 type 1, mean = -1/3.
        # Type 1 has 12 type 0 NNs, therefore variance = 0.
        # variance = 8/12 * (-1 + 1/3)^2 + 4/12 + (1 + 1/3)^2.
        a_var = 2.0 / 3.0 * (-1 + 1.0 / 3.0) ** 2 + 1.0 / 3.0 * (1 + 1.0 /
                                                                 3.0) ** 2
        np_tst.assert_array_almost_equal([0, a_var, a_var, a_var], neigh_var,
                                         decimal=5)
        neigh_var = tool.neighbor_property_variances([-1, 1], 2)
        # Type 0 has 68 type 0 2nd NN and 16 type 1.
        # Type 1 has 48 type 0 2nd NN and 36 type 1
        type0_var = 0.734618916
        type1_var = 0.79338843
        np_tst.assert_array_almost_equal([type1_var, type0_var, type0_var,
                                          type0_var], neigh_var, decimal=5)

    def test_constructor(self):
        # Create a Voronoi tessellation of a BCC crystal.
        bcc = Cell()
        atom = Atom([0, 0, 0], 0)
        bcc.add_atom(atom)
        atom = Atom([0.5, 0.5, 0.5], 0)
        bcc.add_atom(atom)

        # Prepare.
        tool = VoronoiCellBasedAnalysis(radical=True)
        tool.analyze_structure(bcc)

        # Create a new instance based on a B2 crystal.
        b2 = bcc.__copy__()
        b2.get_atom(1).set_type(1)
        tool2 = VoronoiCellBasedAnalysis(old_tessellation=tool,
                                         new_structure=b2)

        # Make sure it balks if I change the basis.
        b2.set_basis(lengths=[2, 1, 1], angles=[90, 90, 90])
        self.assertRaises(Exception.__class__)