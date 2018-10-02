# -*- coding: utf-8 -*-
import unittest
import numpy.testing as np_tst
from chemml.chem.magpie_python import ValenceShellAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry

class testValenceShellAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="CeFeO3")]

        # Make the generator.
        vg = ValenceShellAttributeGenerator()

        # Run the generator.
        features = vg.generate_features(entries)

        # Test results.
        self.assertEqual(4, len(features.columns))
        self.assertEqual(4, features.values[0].size)

        # Results for NaCl.
        np_tst.assert_array_almost_equal([0.375, 0.625, 0, 0],
                                         features.values[0])

        # Results for CeFeO3.
        np_tst.assert_array_almost_equal([2.0 / 6, 2.4 / 6, 1.4 / 6, 0.2 / 6],
                                         features.values[1])