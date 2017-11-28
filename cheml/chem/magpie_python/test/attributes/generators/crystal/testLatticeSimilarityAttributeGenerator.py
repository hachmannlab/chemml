from attributes.generators.crystal.LatticeSimilarityAttributeGenerator import\
    LatticeSimilarityAttributeGenerator
from test.attributes.generators.crystal\
    .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testLatticeSimilarityAttributeGenerator(
    testCoordinationNumberAttributeGenerator):

        def get_generator(self):
            return LatticeSimilarityAttributeGenerator()

        def expected_count(self):
            return 3