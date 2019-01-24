import unittest
import numpy as np
from chemml.search import GeneticAlgorithm

space = ({'alpha': {'uniform': [-2, 0], 
                        'mutation': [0, 2]}}, 
            {'neurons': {'int': [0,10]}},
            {'act': {'choice':[0, 10, 20]}})

count = 0

def evaluate(individual):
    return sum(individual)


class TestGeneticAlgorithm(unittest.TestCase):
    def test_algorithms(self):
        al = [1,2,3,4]
        for i in al:
            ga_search = GeneticAlgorithm(
                evaluate,
                space=space,
                pop_size=20,
                mutation_ratio=0.5,
                algorithm=i)

            best_ind_df, best_individual = ga_search.search(n_generations=10)
            self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 0)
            
    def test_sequential_min(self):
        ga_search = GeneticAlgorithm(evaluate,
                                    fitness="min",
                                    space=space,
                                    pop_size=20,
                                    algorithm=2)

        for _ in range(10):
            best_ind_df, best_individual = ga_search.search(n_generations=1)
        self.assertLessEqual(sum([best_individual[i] for i in best_individual]), 20)
        
    def test_crossovers(self):
        co = ['SinglePoint', 'DoublePoint', 'Blend']
        for c in co:
            ga_search = GeneticAlgorithm(
                evaluate,
                space=space,
                crossover_type=c,
                algorithm=1)
            best_ind_df, best_individual = ga_search.search(n_generations=2)
            self.assertGreaterEqual(sum([best_individual[i] for i in best_individual]), 0)

if __name__ == '__main__':
    unittest.main()
