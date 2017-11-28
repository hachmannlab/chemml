from attributes.generators.crystal.StructuralHeterogeneityAttributeGenerator \
    import StructuralHeterogeneityAttributeGenerator
from test.attributes.generators.crystal\
    .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testStructuralHeterogeneityAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return StructuralHeterogeneityAttributeGenerator()

    def expected_count(self):
        return 8