from chemml.chem.magpie_python import PackingEfficiencyAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testPackingEfficiencyAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return PackingEfficiencyAttributeGenerator()

    def expected_count(self):
        return 1