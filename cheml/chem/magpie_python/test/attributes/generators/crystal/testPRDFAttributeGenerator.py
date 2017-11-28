import unittest
from attributes.generators.crystal.PRDFAttributeGenerator import \
    PRDFAttributeGenerator
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell

class testPRDFAttributeGenerator(unittest.TestCase):
    def test(self):
        # Make two simple structures.
        structure1 = Cell()
        structure1.add_atom(Atom([0, 0, 0], 0))
        structure1.set_type_name(0, "Al")

        entries = []
        entry1 = CrystalStructureEntry(structure1, name="Al", radii=None)
        entries.append(entry1)

        structure2 = Cell()
        structure2.add_atom(Atom([0, 0, 0], 0))
        structure2.set_type_name(0, "Ni")
        structure2.add_atom(Atom([0, 0.5, 0], 1))
        structure2.set_type_name(1, "Al")
        structure2.add_atom(Atom([0, 0, 0.5], 1))
        entry2 = CrystalStructureEntry(structure2, name="NiAl2", radii=None)
        entries.append(entry2)

        # Create feature generator.
        gen = PRDFAttributeGenerator()
        gen.set_cut_off_distance(3.0)
        gen.set_n_points(5)
        gen.set_elements(entries)

        # Add extra element H.
        gen.add_element(name="H")

        # Generate features.
        features = gen.generate_features(entries)

        # Test results.
        self.assertEquals(3 * 3 * 5, features.shape[1])
        self.assertEquals(3 * 3 * 5, len(features.values[0]))
        self.assertAlmostEquals(0, sum(features.values[0][0 : 4 * 5]),
                                delta=1e-6) # First 4 PRDFs are H-X.
        self.assertTrue(max(features.values[0][4 * 5 : 5 * 5]) > 0)
        self.assertAlmostEquals(0, sum(features.values[0][6 * 5 : 9 * 5]),
                                delta=1e-6)  # Only Al in structure.

        self.assertEquals(3 * 3 * 5, len(features.values[1]))
        self.assertAlmostEquals(0, sum(features.values[1][0: 4 * 5]),
                                delta=1e-6)  # First 4 PRDFs are H-X.
        self.assertTrue(max(features.values[1][4 * 5: 5 * 5]) > 0)
        self.assertTrue(max(features.values[1][5 * 5: 6 * 5]) > 0)
        self.assertAlmostEquals(0, sum(features.values[1][6 * 5: 7 * 5]),
                                delta=1e-6)  # Only Al in structure.
        self.assertTrue(max(features.values[1][7 * 5: 8 * 5]) > 0)
        self.assertTrue(max(features.values[1][8 * 5: 9 * 5]) > 0)