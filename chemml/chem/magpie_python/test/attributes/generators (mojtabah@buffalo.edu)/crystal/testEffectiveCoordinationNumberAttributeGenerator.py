from chemml.chem.magpie_python import \
    EffectiveCoordinationNumberAttributeGenerator
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class EffectiveCoordinationNumberAttributeGeneratorTest(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return EffectiveCoordinationNumberAttributeGenerator()