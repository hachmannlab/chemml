from chemml.chem.magpie_python import StructuralHeterogeneityAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    TestCoordinationNumberAttributeGenerator

class TestStructuralHeterogeneityAttributeGenerator(
    TestCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return StructuralHeterogeneityAttributeGenerator()

    def expected_count(self):
        return 8