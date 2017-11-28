from attributes.generators.crystal\
    .EffectiveCoordinationNumberAttributeGenerator import \
    EffectiveCoordinationNumberAttributeGenerator
from test.attributes.generators.crystal\
    .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class EffectiveCoordinationNumberAttributeGeneratorTest(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        return EffectiveCoordinationNumberAttributeGenerator()