from .....attributes.generators.crystal.LatticeSimilarityAttributeGenerator \
    import LatticeSimilarityAttributeGenerator
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator

class testLatticeSimilarityAttributeGenerator(
    testCoordinationNumberAttributeGenerator):

        def get_generator(self):
            return LatticeSimilarityAttributeGenerator()

        def expected_count(self):
            return 3