import unittest

from cheml.search import GA_DEAP

Weights = (1,)
chromosome_length = 2
chromosome_type = (1,1)
bit_limits = ( (0,10), (0,5) )
mut_int_lower = (0,0)
mut_int_upper = (10,5)

def Evaluate(individual):
    return sum(individual),


class TestGA_DEAP(unittest.TestCase):
    def test_algorithm1(self):

        ga_search = GA_DEAP(Evaluate, Weights=Weights, chromosome_length = chromosome_length, chromosome_type = chromosome_type,
                       bit_limits = bit_limits, mut_int_lower=mut_int_lower, mut_int_upper=mut_int_upper )

        ga_search.fit()
        best_ind_df, best_individual = ga_search.algorithm_1()
        self.assertEqual(15, sum(best_individual))

    def test_algorithm2(self):
        ga_search = GA_DEAP(Evaluate, Weights=Weights, chromosome_length=chromosome_length, chromosome_type=chromosome_type,
                            bit_limits=bit_limits, mut_int_lower=mut_int_lower, mut_int_upper=mut_int_upper)
        ga_search.fit()
        best_ind_df, best_individual = ga_search.algorithm_2()
        self.assertEqual(15, sum(best_individual))

    def test_algorithm3(self):
        ga_search = GA_DEAP(Evaluate, Weights=Weights, chromosome_length=chromosome_length, chromosome_type=chromosome_type,
                            bit_limits=bit_limits, mut_int_lower=mut_int_lower, mut_int_upper=mut_int_upper)

        ga_search.fit()
        best_ind_df, best_individual = ga_search.algorithm_3()
        self.assertEqual(15, sum(best_individual))

    def test_algorithm4(self):
        ga_search = GA_DEAP(Evaluate, Weights=Weights, chromosome_length=chromosome_length, chromosome_type=chromosome_type,
                            bit_limits=bit_limits, mut_int_lower=mut_int_lower, mut_int_upper=mut_int_upper)

        ga_search.fit()
        best_ind_df, best_individual = ga_search.algorithm_4()
        self.assertEqual(15, sum(best_individual))


