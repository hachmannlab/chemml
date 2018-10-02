# -*- coding: utf-8 -*-
import unittest
from chemml.chem.magpie_python import IonicityAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry

class testIonicityAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="Al2MnCu"), CompositionEntry(composition="Fe")]

        # Make the generator.
        ig = IonicityAttributeGenerator()

        # Run the generator.
        features = ig.generate_features(entries)

        # Test results.
        self.assertEqual(3, len(features.columns))

        # Results for NaCl.
        self.assertAlmostEqual(1, features.values[0][0], delta=1e-6)
        self.assertAlmostEqual(0.7115, features.values[0][1], delta=1e-2)
        self.assertAlmostEqual(0.3557, features.values[0][2], delta=1e-2)

        # Results for Al2MnCu.
        self.assertAlmostEqual(0, features.values[1][0], delta=1e-6)
        self.assertAlmostEqual(0.0301, features.values[1][1], delta=1e-2)
        self.assertAlmostEqual(0.0092, features.values[1][2], delta=1e-2)

        # Results for Fe.
        self.assertAlmostEqual(0, features.values[2][0], delta=1e-6)
        self.assertAlmostEqual(0, features.values[2][1], delta=1e-2)
        self.assertAlmostEqual(0, features.values[2][2], delta=1e-2)