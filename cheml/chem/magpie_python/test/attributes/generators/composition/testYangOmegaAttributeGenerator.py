import unittest
import os
from attributes.generators.composition.YangOmegaAttributeGenerator import \
    YangOmegaAttributeGenerator
from data.materials.CompositionEntry import CompositionEntry

class testYangOmegaAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="ZrHfTiCuNi"), CompositionEntry(
            composition="CuNi"), CompositionEntry(
            composition="CoCrFeNiCuAl0.3"), CompositionEntry(
            composition="CoCrFeNiCuAl")]

        # Make the generator.
        yg = YangOmegaAttributeGenerator()

        # Run the generator.
        this_file_path = os.path.dirname(__file__)
        rel_path = os.path.join(this_file_path, "../../../../lookup-data/")
        features = yg.generate_features(entries, lookup_path=rel_path)

        # Test results.
        self.assertEquals(2, len(features.columns))

        # Results for ZrHfTiCuNi.
        self.assertAlmostEquals(0.95, features.values[0][0], delta=1e-2)
        self.assertAlmostEquals(0.1021, features.values[0][1], delta=1e-2)

        # Results for CuNi.
        self.assertAlmostEquals(2.22, features.values[1][0], delta=1e-2)
        self.assertAlmostEquals(0.0, features.values[1][1], delta=1e-2)
        # Miracle gives Cu+Ni the same radii as above.

        # Results for CoCrFeNiCuAl0.3.
        # Unable to reproduce paper for CoCrFeNiCuAl0.3, Ward et al. get
        # exactly 1/10th the value of deltaH as reported in the paper. They
        # have repeated this calculation by hand (read: Excel), and believe
        # their result to be correct. They get the same values for deltaS and
        #  T_m. They do get the same value for CoCrFeNiCuAl, so they've
        # concluded this is just a typo in the paper.
        self.assertAlmostEquals(158.5, features.values[2][0], delta=2e-1)
        self.assertAlmostEquals(0.0315, features.values[2][1], delta=1e-2)

        # Results for CoCrFeNiCuAl.
        self.assertAlmostEquals(5.06, features.values[3][0], delta=2e-1)
        self.assertAlmostEquals(0.0482, features.values[3][1], delta=1e-2)