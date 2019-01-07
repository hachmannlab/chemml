import unittest

from chemml.search import GeneticAlgorithm

Weights = (1, )
space = ({'first': {'int': [0, 10]}}, 
        {'second': {'int': [0, 5]}})

def evaluate(individual):
    return sum(individual),


class TestGeneticAlgorithm(unittest.TestCase):
    def test_algorithm1(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            algorithm=1)

        ga_search.initialize()
        best_ind_df, best_individual, final_pop = ga_search.search()
        self.assertGreaterEqual(sum(best_individual), 14)

    def test_algorithm2(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            algorithm=2)

        ga_search.initialize()
        best_ind_df, best_individual, final_pop = ga_search.search()
        self.assertGreaterEqual(sum(best_individual), 14)

    def test_algorithm3(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            algorithm=3)

        ga_search.initialize()
        best_ind_df, best_individual, final_pop = ga_search.search()
        self.assertGreaterEqual(sum(best_individual), 14)


if __name__ == '__main__':
    unittest.main()
