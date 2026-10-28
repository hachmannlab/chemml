from chemml.chem.magpie_python import LatticeSimilarityAttributeGenerator
from test_CoordinationNumberAttributeGenerator import \
    TestCoordinationNumberAttributeGenerator

class TestLatticeSimilarityAttributeGenerator(
    TestCoordinationNumberAttributeGenerator):

        def get_generator(self):
            return LatticeSimilarityAttributeGenerator()

        def expected_count(self):
            return 3