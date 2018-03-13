import unittest
from .....attributes.generators.crystal.CoulombMatrixAttributeGenerator import \
    CoulombMatrixAttributeGenerator
from .....data.materials.CrystalStructureEntry import CrystalStructureEntry
from .....vassal.data.Atom import Atom
from .....vassal.data.Cell import Cell

class testCoulombMatrixAttributeGenerator(unittest.TestCase):
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
        gen = CoulombMatrixAttributeGenerator()
        gen.set_n_eigenvalues(10)

        # Generate features.
        features = gen.generate_features(entries)

        # Test results.
        self.assertEquals(10, features.shape[1])

        self.assertNotAlmostEquals(0, features.values[0][0], delta=1e-6)
        for i in range(1, 10):
            self.assertAlmostEquals(0, features.values[0][i], delta=1e-6)

        self.assertNotAlmostEquals(0, features.values[1][0], delta=1e-6)
        self.assertNotAlmostEquals(0, features.values[1][1], delta=1e-6)
        self.assertNotAlmostEquals(0, features.values[1][2], delta=1e-6)
        for i in range(3, 10):
            self.assertAlmostEquals(0, features.values[1][i], delta=1e-6)