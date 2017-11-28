import unittest
import math
from vassal.analysis.APRDFAnalysis import APRDFAnalysis
import numpy.testing as np_tst
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell

class testAPRDFAnalysis(unittest.TestCase):
    def setUp(self):
        # Make the analysis tool.
        self.tool = APRDFAnalysis()

    def tearDown(self):
        self.tool = None

    def test_eval_distance(self):
        # Define the settings.
        self.tool.set_n_windows(3)
        self.tool.set_cut_off_distance(3.0)

        # Test result.
        np_tst.assert_array_almost_equal([1, 2, 3],
                                         self.tool.get_evaluation_distances())

    def test_APRDF(self):
        # Create a B2 structure with lattice parameter of 1.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0.5], 1))

        self.tool.set_n_windows(2)
        self.tool.set_cut_off_distance(1.0)
        self.tool.set_smoothing_factor(100)

        self.tool.analyze_structure(structure)

        # Trivial: Properties == 0.
        ap_rdf = self.tool.compute_APRDF([0, 0])
        np_tst.assert_array_almost_equal([0.5, 1],
                                         self.tool.get_evaluation_distances())
        np_tst.assert_array_almost_equal([0, 0], ap_rdf)

        # Actual case.
        ap_rdf = self.tool.compute_APRDF([1, 2])

        # Assemble known contributors.
        # [0] -> Number of neighbors * P_i * P_j
        # [1] -> Bond distance
        contributors = []
        contributors.append([2 * 8 * 2 * 1, math.sqrt(3) / 2]) # A-B 1st NN.
        contributors.append([6 * 1 * 1, 1])  # A-A 2nd NN.
        contributors.append([6 * 2 * 2, 1])  # B-B 2nd NN.
        contributors.append([8 * 1 * 1, math.sqrt(3)])  # A-A 3rd NN.
        contributors.append([8 * 2 * 2, math.sqrt(3)])  # B-B 3rd NN.

        eval_dist = [0.5, 1]
        expected_ap_rdf = [sum([c[0] * math.exp(-100 * (c[1] - r)
                        ** 2) for c in contributors]) / 2 for r in eval_dist]
        np_tst.assert_array_almost_equal(expected_ap_rdf, ap_rdf, decimal=3)

    def test_unit_cell_choice(self):
        # Create a B2 structure with lattice parameter of 1.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0.5], 1))

        # Create a 2x1x1 supercell.
        supercell = Cell()
        supercell.set_basis(lengths=[2, 1, 1], angles=[90, 90, 90])
        supercell.add_atom(Atom([0, 0, 0], 0))
        supercell.add_atom(Atom([0.5, 0, 0], 0))
        supercell.add_atom(Atom([0.25, 0.5, 0.5], 1))
        supercell.add_atom(Atom([0.75, 0.5, 0.5], 1))

        self.tool.set_cut_off_distance(3.0)
        self.tool.set_n_windows(10)
        self.tool.set_smoothing_factor(4)


        # Compute the primitive cell AP-RDF.
        self.tool.analyze_structure(structure)
        p_ap_rdf = self.tool.compute_APRDF([1, 2])

        # Compute the supercell AP-RDF.
        self.tool.analyze_structure(supercell)
        sc_ap_rdf = self.tool.compute_APRDF([1, 2])

        # Compare results.
        np_tst.assert_array_almost_equal(p_ap_rdf, sc_ap_rdf)