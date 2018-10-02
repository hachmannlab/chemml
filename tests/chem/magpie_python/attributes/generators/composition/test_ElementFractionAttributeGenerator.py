# -*- coding: utf-8 -*-
import unittest
from chemml.chem.magpie_python import ElementFractionAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry
from chemml.chem.magpie_python.data.materials.util.LookUpData import LookUpData

class testElementFractionAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="Fe")]

        # Make generator.
        el = ElementFractionAttributeGenerator()

        # Run generator.
        features = el.generate_features(entries)

        # Test results.
        self.assertEqual(len(LookUpData.element_names), features.values[
            0].size)
        self.assertAlmostEqual(1.0, sum(features.values[0]), delta=1e-6)
        self.assertAlmostEqual(0.0, min(features.values[0]), delta=1e-6)
        self.assertAlmostEqual(0.5, features.values[0][10], delta=1e-6)
        self.assertAlmostEqual(1.0, sum(features.values[1]), delta=1e-6)
        self.assertAlmostEqual(0.0, min(features.values[1]), delta=1e-6)
        self.assertAlmostEqual(1.0, features.values[1][25], delta=1e-6)