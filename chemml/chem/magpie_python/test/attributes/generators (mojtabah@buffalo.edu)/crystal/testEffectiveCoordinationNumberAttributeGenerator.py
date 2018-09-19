from .....attributes.generators.crystal\
    .EffectiveCoordinationNumberAttributeGenerator import \
    EffectiveCoordinationNumberAttributeGenerator
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class EffectiveCoordinationNumberAttributeGeneratorTest(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return EffectiveCoordinationNumberAttributeGenerator()