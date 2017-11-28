from attributes.generators.crystal.PackingEfficiencyAttributeGenerator import \
    PackingEfficiencyAttributeGenerator
from test.attributes.generators.crystal\
    .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testPackingEfficiencyAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return PackingEfficiencyAttributeGenerator()

    def expected_count(self):
        return 1