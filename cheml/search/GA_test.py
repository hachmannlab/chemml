from genetic_algorithm import GA_DEAP

Weights = (1,)
chromosome_length = 2
chromosome_type = (1,1)
bit_limits = ( (0,10), (0,5) )
mut_int_lower = (0,0)
mut_int_upper = (10,5)

def Evaluate(individual):
    return sum(individual),

ga_search = GA_DEAP(Evaluate, Weights=Weights, chromosome_length = chromosome_length, chromosome_type = chromosome_type,
               bit_limits = bit_limits, mut_int_lower=mut_int_lower, mut_int_upper=mut_int_upper )

ga_search.fit()
best_ind_df, best_individual = ga_search.algorithm_1()
#best_ind_df, best_individual = ga_search.algorithm_2()
#best_ind_df, best_individual = ga_search.algorithm_3()
#best_ind_df, best_individual = ga_search.algorithm_4()

