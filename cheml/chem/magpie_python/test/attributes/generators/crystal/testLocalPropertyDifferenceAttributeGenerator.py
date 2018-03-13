import numpy as np
import numpy.testing as np_tst
from .....attributes.generators.crystal\
    .LocalPropertyDifferenceAttributeGenerator import \
    LocalPropertyDifferenceAttributeGenerator
from .....data.materials.CrystalStructureEntry import CrystalStructureEntry
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator
from .....vassal.data.Atom import Atom
from .....vassal.data.Cell import Cell

class testLocalPropertyDifferenceAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        gen = LocalPropertyDifferenceAttributeGenerator()
        gen.clear_shells()
        gen.add_shells([1, 2])
        gen.add_elemental_property("Electronegativity")
        return gen

    def expected_count(self):
        return 10

    def test_results2(self):
        # Create a B1-HHe structure.
        structure = Cell()
        basis = np.zeros((3, 3))
        basis[0] = np.array([0, 0.5, 0.5])
        basis[1] = np.array([0.5, 0, 0.5])
        basis[2] = np.array([0.5, 0.5, 0])
        structure.set_basis(basis=basis)
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0.5], 1))
        structure.set_type_name(0, "H")
        structure.set_type_name(1, "He")

        entries = [CrystalStructureEntry(structure, name="B1-HHe", radii=None)]

        # Get the feature generator.
        gen = self.get_generator()
        gen.clear_elemental_properties()
        gen.add_elemental_property("Number")

        # Generate features.
        features = gen.generate_features(entries)

        # Test the results.
        self.assertEquals(self.expected_count(), features.shape[1])
        np_tst.assert_array_almost_equal([1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                         features.values[0])