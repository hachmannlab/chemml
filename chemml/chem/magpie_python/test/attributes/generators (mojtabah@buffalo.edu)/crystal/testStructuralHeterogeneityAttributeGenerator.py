from chemml.chem.magpie_python import StructuralHeterogeneityAttributeGenerator
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testStructuralHeterogeneityAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return StructuralHeterogeneityAttributeGenerator()

    def expected_count(self):
        return 8