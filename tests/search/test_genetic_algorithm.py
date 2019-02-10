import unittest
from chemml.search import GeneticAlgorithm

space = ({'alpha': {'uniform': [-20, 0], 
                        'mutation': [0, 2]}}, 
            {'neurons': {'int': [0,10]}},
            {'act': {'choice':range(0,100,5)}})


def evaluate(individual):
    return sum(individual)


class TestGeneticAlgorithm(unittest.TestCase):
    def test_algorithms(self):
        al = [3]
        for i in al:
            ga_search = GeneticAlgorithm(
                evaluate,
                space=space,
                pop_size=10,
                mutation_size=0.4,
                crossover_size=0.6,
                algorithm=i)

            best_ind_df, best_individual = ga_search.search(n_generations=4)
            self.assertLessEqual(sum([best_individual[i] for i in best_individual]), 200)
            
    def test_sequential_min(self):
        ga_search = GeneticAlgorithm(evaluate,
                                    fitness="min",
                                    space=space,
                                    pop_size=10,
                                    algorithm=3)

        for _ in range(4):
            best_ind_df, best_individual = ga_search.search(n_generations=1)
        self.assertLessEqual(sum([best_individual[i] for i in best_individual]), 200)
        
    def test_crossovers(self):
        co = ['SinglePoint', 'DoublePoint', 'Blend']
        for c in co:
            ga_search = GeneticAlgorithm(
                evaluate,
                space=space,
                crossover_type=c,
                algorithm=3)
            best_ind_df, best_individual = ga_search.search(n_generations=4)
            self.assertLessEqual(sum([best_individual[i] for i in best_individual]), 200)

if __name__ == '__main__':
    unittest.main()
