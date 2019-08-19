import pytest
from chemml.optimization import GeneticAlgorithm

space = ({'alpha': {'uniform': [-20, 0], 
                        'mutation': [0, 2]}}, 
            {'neurons': {'int': [0,10]}},
            {'act': {'choice':range(0,100,5)}})


def evaluate(individual):
    return sum(individual)


def test_algorithms():
    al = [3]
    for i in al:
        ga_search = GeneticAlgorithm(
            evaluate,
            space=space,
            pop_size=10,
            mutation_size=4,
            crossover_size=4,
            algorithm=i)

        _, best_individual = ga_search.search(n_generations=4)
        assert sum([best_individual[i] for i in best_individual]) <= 200
        
def test_sequential_min():
    ga_search = GeneticAlgorithm(evaluate,
                                fitness=("min", ),
                                space=space,
                                pop_size=10,
                                mutation_size=5,
                                crossover_size=5,
                                algorithm=3)

    for _ in range(4):
        _, best_individual = ga_search.search(n_generations=1)
    assert sum([best_individual[i] for i in best_individual]) <= 200
    
def test_crossovers():
    co = ['SinglePoint', 'DoublePoint', 'Blend']
    for c in co:
        ga_search = GeneticAlgorithm(
            evaluate,
            space=space,
            crossover_type=c,
            pop_size=10,
            mutation_size=4,
            crossover_size=4,
            algorithm=3)
        _, best_individual = ga_search.search(n_generations=4)
        assert sum([best_individual[i] for i in best_individual]) <= 200

