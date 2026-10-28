from chemml.chem.magpie_python import \
    EffectiveCoordinationNumberAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    TestCoordinationNumberAttributeGenerator

class EffectiveCoordinationNumberAttributeGeneratorTest(
    TestCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return EffectiveCoordinationNumberAttributeGenerator()