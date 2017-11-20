import unittest
import numpy.testing as np_tst
import os
from attributes.generators.composition.ValenceShellAttributeGenerator import \
    ValenceShellAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testValenceShellAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="CeFeO3")]

        # Make the generator.
        vg = ValenceShellAttributeGenerator()

        # Run the generator.
        this_file_path = os.path.dirname(__file__)
        rel_path = os.path.join(this_file_path, "../../../../lookup-data/")
        features = vg.generate_features(entries, lookup_path=rel_path)

        # Test results.
        self.assertEquals(4, len(features.columns))
        self.assertEquals(4, features.values[0].size)

        # Results for NaCl.
        np_tst.assert_array_almost_equal([0.375, 0.625, 0, 0],
                                         features.values[0])

        # Results for CeFeO3.
        np_tst.assert_array_almost_equal([2.0 / 6, 2.4 / 6, 1.4 / 6, 0.2 / 6],
                                         features.values[1])