import unittest
import math
import numpy.testing as np_tst
from .....attributes.generators.crystal.APRDFAttributeGenerator import \
    APRDFAttributeGenerator
from .....data.materials.CrystalStructureEntry import CrystalStructureEntry
from .....vassal.data.Atom import Atom
from .....vassal.data.Cell import Cell

class testAPRDFAttributeGenerator(unittest.TestCase):
    def test(self):
        structure = Cell()
        structure.set_basis(lengths=[3.2, 3.2, 3.2], angles=[90, 90, 90])
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0.5], 1))
        structure.set_type_name(0, "Ni")
        structure.set_type_name(1, "Al")

        entry = CrystalStructureEntry(structure, name="B2-NiAl", radii=None)
        entries = [entry]

        # Create feature generator.
        gen = APRDFAttributeGenerator()
        gen.set_cut_off_distance(3.2)
        gen.set_num_points(2)
        gen.set_smoothing_parameter(100)
        gen.add_elemental_property("Number")

        # Generate features.
        features = gen.generate_features(entries)
        self.assertEquals(2, len(features.columns))

        ap_rdf = features.values

        # Assemble known contributors.
        # [0] -> Number of neighbors * P_i * P_j
        # [1] -> Bond distance
        contributors = []
        contributors.append([2 * 8 * 13 * 28, 3.2 * math.sqrt(3) / 2])  # A-B
        #  1st NN.
        contributors.append([6 * 13 * 13, 3.2 * 1])  # A-A 2nd NN.
        contributors.append([6 * 28 * 28, 3.2 * 1])  # B-B 2nd NN.
        contributors.append([8 * 13 * 13, 3.2 * math.sqrt(3)])  # A-A 3rd NN.
        contributors.append([8 * 28 * 28, 3.2 * math.sqrt(3)])  # B-B 3rd NN.

        eval_dist = [1.6, 3.2]
        expected_ap_rdf = [sum([c[0] * math.exp(-100 * (c[1] - r)
                                                ** 2) for c in
                                contributors]) / 2 for r in eval_dist]
        np_tst.assert_array_almost_equal(expected_ap_rdf, ap_rdf[0])