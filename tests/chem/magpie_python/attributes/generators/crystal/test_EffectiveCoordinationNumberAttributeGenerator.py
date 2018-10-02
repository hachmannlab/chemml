from chemml.chem.magpie_python import \
    EffectiveCoordinationNumberAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class EffectiveCoordinationNumberAttributeGeneratorTest(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return EffectiveCoordinationNumberAttributeGenerator()