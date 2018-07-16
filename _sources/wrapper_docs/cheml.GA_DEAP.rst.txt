.. _GA_DEAP:

GA_DEAP
========

:task:
    | Search

:subtask:
    | genetic algorithm

:host:
    | cheml

:function:
    | GA_DEAP

:input tokens (receivers):
    | ``evaluate`` : a function that receives a list of individuals and returns the score
    |   types: ("<type 'function'>",)

:output tokens (senders):
    | ``best_individual`` : pandas dataframe of the best individual
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``best_ind_df`` : pandas dataframe of best individuals after each iteration
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``func_method`` : string, (default:algorithm_1)
    |   a method of the GA_DEAP class that should be applied
    |   choose one of: ('algorithm_1', 'algorithm_2', 'algorithm_3', 'algorithm_4')

:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | deap, 1.2.2

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = GA_DEAP``
    |   ``<< func_method = algorithm_1``
    |   ``<< mut_float_dev = 1``
    |   ``<< init_pop_frac = 0.35``
    |   ``<< crossover_type = Blend``
    |   ``<< chromosome_type = (1,)``
    |   ``<< n_generations = 20``
    |   ``<< Evaluate = @evaluate``
    |   ``<< chromosome_length = 1``
    |   ``<< crossover_pop_frac = 0.35``
    |   ``<< crossover_prob = 0.4``
    |   ``<< Weights = (-1.0,)``
    |   ``<< mut_float_mean = 0``
    |   ``<< bit_limits = ((0, 10),)``
    |   ``<< mut_int_lower = (1,)``
    |   ``<< mut_int_upper = (10,)``
    |   ``<< pop_size = 50``
    |   ``<< mutation_prob = 0.4``
    |   ``>> id evaluate``
    |   ``>> id best_individual``
    |   ``>> id best_ind_df``
    |
    .. note:: The documentation page for function parameters: 
