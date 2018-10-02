# -*- coding: utf-8 -*-
import unittest
from math import sqrt
import numpy.testing as np_tst
from chemml.chem.magpie_python import  StoichiometricAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry

class testStoichiometricAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl")]

        # Make the generator and set options.
        sg = StoichiometricAttributeGenerator(use_default_norms=False)
        sg.add_p_norm(2)
        sg.add_p_norm(3)

        # Run the generator.
        features = sg.generate_features(entries)

        # Test results.
        self.assertEqual(3, len(features.columns))
        self.assertEqual(3, features.values[0].size)
        np_tst.assert_array_almost_equal([2, sqrt(0.5), 0.25 ** (1.0 / 3)],
                                         features.values[0])