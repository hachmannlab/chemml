import unittest

from chemml.search import GeneticAlgorithm

Weights = (1, )
chromosome_length = 2
chromosome_type = (1, 1)
bit_limits = ((0, 10), (0, 5))
mut_int_lower = (0, 0)
mut_int_upper = (10, 5)


def evaluate(individual):
    return sum(individual),


class TestGeneticAlgorithm(unittest.TestCase):
    def test_algorithm1(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            chromosome_length=chromosome_length,
            chromosome_type=chromosome_type,
            bit_limits=bit_limits,
            mut_int_lower=mut_int_lower,
            mut_int_upper=mut_int_upper,
            algorithm=1)

        ga_search.fit()
        best_ind_df, best_individual = ga_search.search()
        self.assertEqual(15, sum(best_individual))

    def test_algorithm2(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            chromosome_length=chromosome_length,
            chromosome_type=chromosome_type,
            bit_limits=bit_limits,
            mut_int_lower=mut_int_lower,
            mut_int_upper=mut_int_upper,
            algorithm=2)
        ga_search.fit()
        best_ind_df, best_individual = ga_search.search()
        self.assertEqual(15, sum(best_individual))

    def test_algorithm3(self):
        ga_search = GeneticAlgorithm(
            evaluate,
            weights=Weights,
            chromosome_length=chromosome_length,
            chromosome_type=chromosome_type,
            bit_limits=bit_limits,
            mut_int_lower=mut_int_lower,
            mut_int_upper=mut_int_upper,
            algorithm=3)

        ga_search.fit()
        best_ind_df, best_individual = ga_search.search()
        self.assertEqual(15, sum(best_individual))


if __name__ == '__main__':
    unittest.main()
