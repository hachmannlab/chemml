import unittest
from attributes.generators.composition.ElementFractionAttributeGenerator \
    import ElementFractionAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry
from data.materials.util.LookUpData import LookUpData

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
        self.assertEquals(len(LookUpData.element_names), features.values[
            0].size)
        self.assertAlmostEquals(1.0, sum(features.values[0]), delta=1e-6)
        self.assertAlmostEquals(0.0, min(features.values[0]), delta=1e-6)
        self.assertAlmostEquals(0.5, features.values[0][10], delta=1e-6)
        self.assertAlmostEquals(1.0, sum(features.values[1]), delta=1e-6)
        self.assertAlmostEquals(0.0, min(features.values[1]), delta=1e-6)
        self.assertAlmostEquals(1.0, features.values[1][25], delta=1e-6)