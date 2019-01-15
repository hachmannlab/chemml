import unittest

from chemml.search import GeneticAlgorithm


Weights = (1, )
space = ({'first': {'int': [0, 10]}}, 
        {'second': {'int': [0, 5]}})
count = 0

def evaluate(individual):
    global count
    count += 1
    return sum(individual),


class TestGeneticAlgorithm(unittest.TestCase):
    def test_algorithm1(self):
        global count
        count = 0
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            pop_size=10,
            algorithm=1)

        ga_search.initialize()
        best_ind_df, best_individual = ga_search.search(n_generations=2)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)
        self.assertEqual(count, 43)

    def test_algorithm2(self):
        global count
        count = 0
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            pop_size=10,
            algorithm=2)

        ga_search.initialize()
        best_ind_df, best_individual = ga_search.search(n_generations=2)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)
        self.assertEqual(count, 43)

    def test_sequential_runs(self):
        global count
        count = 0
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            pop_size=10,
            algorithm=3)

        ga_search.initialize()
        for _ in range(2):
            best_ind_df, best_individual = ga_search.search(n_generations=1)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)
        self.assertEqual(count, 64)

    def test_onepoint(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            crossover_type="OnePoint",
            algorithm=1)
        ga_search.initialize()
        best_ind_df, best_individual = ga_search.search(n_generations=2)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)

    def test_twopoint(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            crossover_type="TwoPoint",
            algorithm=2)
        ga_search.initialize()
        best_ind_df, best_individual = ga_search.search(n_generations=2)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)

    def test_blend(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            space=space,
            crossover_type="Blend",
            algorithm=3)
        ga_search.initialize()
        best_ind_df, best_individual = ga_search.search(n_generations=2)
        self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 14)

if __name__ == '__main__':
    unittest.main()
