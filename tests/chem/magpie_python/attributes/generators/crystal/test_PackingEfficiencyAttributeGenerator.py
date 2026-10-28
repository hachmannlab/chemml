from chemml.chem.magpie_python import PackingEfficiencyAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    TestCoordinationNumberAttributeGenerator

class TestPackingEfficiencyAttributeGenerator(
    TestCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return PackingEfficiencyAttributeGenerator()

    def expected_count(self):
        return 1