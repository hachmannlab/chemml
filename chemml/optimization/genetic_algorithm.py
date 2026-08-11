import random
import pandas as pd
import time
import math
import numpy as np
import multiprocessing as mp
import pickle
import os
from copy import deepcopy
import itertools


def _evaluate_individual_mp(args):
    evaluate, individual = args
    return evaluate(individual)

class GeneticAlgorithm(object):
    """
    A python implementation of real-valued, genetic algorithm for solving optimization problems.

    Parameters
    ----------
    evaluate : function
        The objective function that has to be optimized. The first parameter of the objective function is a list of the trial values of the hyper-parameters in the order in which they are declared in the space variable. The objective function should always return a tuple with the metric/metrics for single/multi-objective optimization.

    space : tuple,
        A tuple of dict objects specifying the hyper-parameter space to search in.
        Each hyper-parameter should be a python dict object with the name of the hyper-parameter as the key.
        Value is also a dict object with one mandatory key among: 'uniform', 'int' and 'choice' for defining floating point, integer and choice variables respectively.
        Values for these keys should be a list defining the valid hyper-parameter search space (lower and upper bounds for 'int' and 'uniform', and all valid choices for 'choice').
        For uniform, a 'mutation' key is also required for which the value is [mean, standard deviation] for the gaussian distribution.
        Example:
                ({'alpha': {'uniform': [0.001, 1],
                            'mutation': [0, 1]}},
                {'layers': {'int': [1, 3]}},
                {'neurons': {'choice': range(0,200,20)}})

    fitness : tuple, optional (default = ('Max',)
        A tuple of string(s) for Maximizing (Max) or minimizing (Min) the objective function(s).

    pop_size : integer, optional (default = 50)
        Size of the population

    crossover_size : int, optional (default = 30)
        Number of individuals to select for crossover.

    mutation_size : int, optional (default = 20)
        Number of individuals to select for mutation.

    crossover_type : string, optional (default = "Blend")
        Type of crossover: SinglePoint, DoublePoint, Blend, Uniform

    mutation_prob : float, optional (default = 0.4)
        Probability of mutation.

    algorithm : int, optional (default=1)
        The algorithm to use for the search. Look at the 'search' method for a description of the various algorithms.

            - Algorithm 1:
                Initial population is instantiated.
                Roulette wheel selection is used for selecting individuals for crossover and mutation.
                The initial population, crossovered and mutated individuals form the pool of individuals from which the best
                n members are selected as the initial population for the next generation, where n is the size of population.

            - Algorithm 2:
                Same as algorithm 1 but when selecting individuals for next generation, n members are selected using Roulette wheel selection.

            - Algorithm 3:
                Same as algorithm 1 but when selecting individuals for next generation, best members from each of the three pools (initital population, crossover and mutation) are selected according to the input parameters in the search method.

            - Algorithm 4:
                Same as algorithm 1 but mutation population is selected from the crossover population and not from the parents directly.

    initial_population : list, optional (default=None)
        The initial population for the algorithm to start with. If not provided, initial population is randomly generated.

    active_fraction : float, optional (default=None) (ONLY FOR FEATURE SELECTION)
        Target probability of selecting 1 for binary choice genes during large-space fallback sampling
        (only when there are no uniform variables and chromosome length is greater than 20).

    target_features_count : int, optional (default=None) (ONLY FOR FEATURE SELECTION)
        Target number of active (value 1) binary choice genes during large-space fallback sampling.
        This is converted internally to ``active_fraction = target_features_count / n_binary_genes``.

    """

    def __init__(self, 
                evaluate, 
                space,
                fitness=("Max", ), 
                pop_size=50,
                crossover_size=30,
                mutation_size=20,
                crossover_type="Blend",
                fused_cutoff = 5,
                mutation_prob=0.6,
                algorithm=3,
                initial_population=None,
                n_jobs=1,
                active_fraction=None,
                target_features_count=None):

        self.chromosome_length = len(space)
        if self.chromosome_length < 1:
            print("Space variable not defined. Aborting.")
            exit(code=1)
        if self.chromosome_length <2 and crossover_type == "SinglePoint": raise Exception('Single point crossover not possible for chromosome length 1.')
        if self.chromosome_length <3 and crossover_type == "DoublePoint": raise Exception('Double point crossover not possible for chromosome length 2.')
        self.chromosome_type, self.bit_limits, self.mutation_params, self.var_names = [], [], [], []
        
        uni = 0
        for param_dict in space:
            for name in param_dict:
                self.var_names.append(name)
                var = param_dict[name]
                if 'uniform' in var:
                    self.chromosome_type.append('uniform')
                    self.bit_limits.append(var['uniform'])
                    uni += 1

                elif 'int' in var:
                    self.chromosome_type.append('int')
                    self.bit_limits.append(var['int'])

                elif 'choice' in var:
                    self.chromosome_type.append('choice')
                    self.bit_limits.append(var['choice'])
                
                if 'mutation' in var:
                    self.mutation_params.append(var['mutation'])
                else: self.mutation_params.append(None)
        
        self.n_jobs = n_jobs
        self.evaluate = evaluate
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_type = crossover_type
        self.fused_cutoff = fused_cutoff
        self.crossover_size = crossover_size
        self.mutation_size = mutation_size
        self.algo = algorithm
        self.initial_pop = initial_population
        self.active_fraction = active_fraction
        self.target_features_count = target_features_count
        self.fit_val, self.population, self.fitness_dict, self.global_cm_list = [], None, {}, None
        self._fitness_validated = False
        for i in fitness:
            if i.lower() == 'max': self.fit_val.append(1)
            elif i.lower() == 'min': self.fit_val.append(-1)
            else: raise Exception("Fitness value should be either 'max' or 'min'.")

        # For feature selection; validate the active_fraction and target_features_count inputs if provided, and ensure they are compatible with the space parameter.
        if self.active_fraction is not None or self.target_features_count is not None:
            self._validate_feature_bias_inputs(uni)

        # if there is no uniform type in the space parameter, use a pre-defined crossover and mutation list as follows: 
        if uni == 0:
            if len(space) == 1: self.global_cm_list = self.bit_limits[0]
            else:
                gcl = []
                for i, t in zip(self.bit_limits, self.chromosome_type):
                    if t == 'int':
                        gcl.append(list(range(i[0], i[1]+1)))
                    else: gcl.append(i)
                if len(gcl) <= 20:
                    self.global_cm_list = list(itertools.product(*gcl))
                else:
                    active_prob = self._get_active_probability_for_large_space()
                    self.global_cm_list = [
                        tuple([self._sample_global_cm_value(sublist, active_prob) for sublist in gcl])
                        for _ in range(len(gcl)**2)
                    ]

    def _validate_fitness_dimensions(self):
        # Validate that fitness tuple length matches the number of objectives from evaluate
        if self._fitness_validated or not self.fitness_dict:
            return
        self._fitness_validated = True
        
        first_fitness = next(iter(self.fitness_dict.values()))
        if not isinstance(first_fitness, (tuple, list)):
            first_fitness = (first_fitness,)
        
        n_objectives = len(first_fitness)
        n_directions = len(self.fit_val)
        
        if n_objectives == n_directions:
            return
        
        if n_objectives == 1 and n_directions > 1:
            direction_name = "Max" if self.fit_val[0] == 1 else "Min"
            import warnings
            warnings.warn(
                f"Only one output returned from evaluate(), but {n_directions} directions "
                f"specified in fitness tuple. Only the first direction ('{direction_name}') will be used.",
                UserWarning,
                stacklevel=4
            )
            self.fit_val = self.fit_val[:1]
        
        elif n_objectives > 1 and n_directions == 1:
            direction_name = "Max" if self.fit_val[0] == 1 else "Min"
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Only one direction ('{direction_name}') specified in fitness tuple, but "
                f"{n_objectives} objectives returned from evaluate(). All objectives will be "
                f"optimized in that direction."
            )
        
        else:
            raise AttributeError(
                f"Mismatch between fitness dimensions: {n_directions} directions specified "
                f"in fitness tuple, but evaluate() returns {n_objectives} objectives. "
                f"Please ensure the fitness tuple has the same length as the number of values "
                f"returned by the evaluate function."
            )

    def _is_binary_choice(self, values):
        return len(values) == 2 and set(values) == set([0, 1])

    def _validate_feature_bias_inputs(self, uni):
        if self.active_fraction is not None and self.target_features_count is not None:
            raise ValueError("Both active_fraction and target_features_count were provided. Please provide only one.")

        if uni != 0:
            raise ValueError("Feature-bias parameters are only supported when there are no uniform variables in space.")

        if self.chromosome_length <= 20:
            raise ValueError("Feature-bias parameters are only supported for large-space fallback sampling (chromosome length > 20).")

        invalid_binary = [
            limits for ctype, limits in zip(self.chromosome_type, self.bit_limits)
            if ctype != 'choice' or not self._is_binary_choice(limits)
        ]
        if invalid_binary:
            raise ValueError("Feature-bias parameters are only supported for pure 'choice' spaces with values [0, 1].")

        if self.active_fraction is not None:
            if isinstance(self.active_fraction, bool) or not isinstance(self.active_fraction, (int, float)):
                raise TypeError("active_fraction must be a float in [0, 1].")
            if not 0.0 <= float(self.active_fraction) <= 1.0:
                raise ValueError("active_fraction must be in [0, 1].")

        if self.target_features_count is not None:
            if isinstance(self.target_features_count, bool) or not isinstance(self.target_features_count, int):
                raise TypeError("target_features_count must be an integer.")
            if self.target_features_count < 0:
                raise ValueError("target_features_count must be >= 0.")
            n_binary = sum(1 for limits in self.bit_limits if self._is_binary_choice(limits))
            if self.target_features_count > n_binary:
                raise ValueError("target_features_count cannot be greater than the number of binary genes.")

    def _get_active_probability_for_large_space(self):
        if self.active_fraction is None and self.target_features_count is None:
            return None
        if self.active_fraction is not None:
            return float(self.active_fraction)
        target_features_count = self.target_features_count
        if target_features_count is None:
            raise ValueError("target_features_count is required when active_fraction is not provided.")
        n_binary = sum(1 for limits in self.bit_limits if self._is_binary_choice(limits))
        if n_binary == 0:
            raise ValueError("No binary genes found for target_features_count.")
        return float(target_features_count) / float(n_binary)

    def _sample_global_cm_value(self, sublist, active_prob):
        if active_prob is not None and self._is_binary_choice(sublist):
            return 1 if random.random() < active_prob else 0
        return random.choice(sublist)

    def pop_generator(self, n):
        pop = []
        if self.initial_pop is not None:
            for i in self.initial_pop:
                pop.append(tuple(i))
        else:
            for x in range(n):
                while True:
                    new = self.chromosome_generator(x, n)
                    if new not in pop:
                        pop.append(new)
                        break
        return pop

    def chromosome_generator(self, x, n):
        chsome = []
        for i, j in zip(self.bit_limits, self.chromosome_type):
            if j == 'uniform':
                chsome.append(np.linspace(i[0], i[1], n)[x])
            elif j == 'int':
                chsome.append(random.randint(i[0], i[1]))
            elif j == 'choice':
                chsome.append(random.choice(i))
        return tuple(chsome)

    def SinglePointCrossover(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        nVar=len(x1)
        c = random.randint(1,nVar-1)
        y1=x1[0:c]				
        y1=y1+x2[c:nVar]      
        y2=x2[0:c]
        y2=y2+x1[c:nVar]
        return tuple(deepcopy(y1)), tuple(deepcopy(y2))

    def DoublePointCrossover(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        nVar = len(x1)
        cc = random.sample(range(1,nVar), 2)   
        c1 = min(cc)
        c2 = max(cc)
        y1 = x1[0:c1]+x2[c1:c2]+x1[c2:nVar]				
        y2 = x2[0:c1]+x1[c1:c2]+x2[c2:nVar]      
        return tuple(deepcopy(y1)), tuple(deepcopy(y2))

    def UniformCrossover(self, x1, x2):
        parents = [x1,x2]
        ind1, ind2 = [parents[random.randint(0, 1)][i] for i in range(len(x1))], [parents[random.randint(0, 1)][i] for i in range(len(x1))]   
        return tuple(deepcopy(ind1)), tuple(deepcopy(ind2))

    def blend(self, ind1, ind2, fitness_dict, z=0.4, alpha=0.5, beta=0.1):
        # rank all individuals in fitness dict
        ranked_list = self.select([i for i in fitness_dict], fitness_dict, len(fitness_dict.items()), choice="best")
        # determine the better individual among the two for implementing the alpha-beta crossover
        if ranked_list.index(ind1) < ranked_list.index(ind2): better, worse = deepcopy(list(ind1)), deepcopy(list(ind2))
        else: better, worse = deepcopy(list(ind2)), deepcopy(list(ind1))
        
        for i in range(self.chromosome_length):
            # for type 'choice' apply roulette wheel selection by biasing the wheel towards the two current values
            if self.chromosome_type[i] == 'choice':
                if len(self.bit_limits[i]) == 2: better[i], worse[i] = worse[i], better[i]
                else:
                    scores = []
                    for inx, ch in enumerate(self.bit_limits[i]):
                        if ch == better[i] or ch == worse[i]: scores.append(3)
                        else: scores.append(1)
                    sc_dict = {self.bit_limits[i][j]: scores[j] for j in range(len(scores))}
                    better[i], worse[i] = self.select(self.bit_limits[i], sc_dict, 2)
            else:
                while True:
                    d = abs(better[i] - worse[i])
                    if (better[i] <= worse[i]):
                        min = better[i] - d * alpha
                        max = worse[i] + d * beta
                    else:
                        min = worse[i] - d * beta
                        max = better[i] + d * alpha
                    tm1, tm2 = min + random.random() * (max - min), min + random.random() * (max - min)
                    # check new values with user-defined bounds
                    if self.bit_limits[i][0] <= tm1 <= self.bit_limits[i][1] and self.bit_limits[i][0] <= tm2 <= self.bit_limits[i][1]: break
                better[i], worse[i] = tm1, tm2
                if self.chromosome_type[i] == 'int':
                    better[i], worse[i] = int(better[i]), int(worse[i])
        return tuple(deepcopy(better)), tuple(deepcopy(worse))

    def fused(self, ind1, ind2, fitness_dict):
        ind1, ind2 = list(ind1), list(ind2)
        x_ind1, y_ind1 = ind1[:self.fused_cutoff], ind1[self.fused_cutoff:]
        x_ind2, y_ind2 = ind2[:self.fused_cutoff], ind2[self.fused_cutoff:]
        x_ind1, x_ind2 = self.blend(x_ind1, x_ind2, fitness_dict)
        y_ind1, y_ind2 = self.UniformCrossover(y_ind1, y_ind2)
        return tuple(deepcopy(list(x_ind1) + list(y_ind1))), tuple(deepcopy(list(x_ind2) + list(y_ind2)))

    def select(self, population, fit_dict, num, choice="Roulette"):
        if num >= len(population): return population
        o_fits = [fit_dict[i] for i in population]

        df_fits = pd.DataFrame(o_fits)
        # scale all values in range 1-2
        df2 = [((df_fits[i] - df_fits[i].min()) / (df_fits[i].max() - df_fits[i].min())) + 1 for i in range(df_fits.shape[1])]
        # inverse min columns
        df2 = pd.DataFrame([df2[i]**self.fit_val[i] for i in range(len(df2))]).T
        # rescale all values in range 1-2
        df2 = pd.DataFrame([((df2[i] - df2[i].min()) / (df2[i].max() - df2[i].min())) + 1 for i in range(df2.shape[1])])
        
        fitnesses = list(df2.sum())

        if choice == "Roulette":
            total_fitness = float(sum(fitnesses))
            rel_fitness = [f/total_fitness for f in fitnesses]
            # Generate probability intervals for each individual
            probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
            # Draw new population
            new_population = []
            for _ in range(num):
                r = random.random()
                for i, individual in enumerate(population):
                    if r <= probs[i]:
                        new_population.append(deepcopy(individual))
                        break
            return new_population
        else:
            fits_sort = sorted(fitnesses, reverse=True)
            best = [deepcopy(population[fitnesses.index(fits_sort[i])]) for i in range(min(num, len(population)))]
            return best

    def custom_mutate(self, indi, fitness_dict):
        # rank all individuals in fitness dict
        ranked_list = self.select([i for i in fitness_dict], fitness_dict, len(fitness_dict.items()), choice="best")
        # calculate parameter to adjust Gaussian distribution according to individual's rank for uniform type
        parent_fit_param = (ranked_list.index(indi) + 1)*2/len(ranked_list)
        indi = list(indi)
        # if there is no uniform type hyperparamter in the space variable, run the 'if' condition below to select from a pre-defined superlist of mutations and crossovers.
        if self.global_cm_list is not None:
            try:
                if self.chromosome_length == 1:
                    indi = random.choice(list(set(self.global_cm_list)-set([i[0] for i in fitness_dict.keys()])))
                    if isinstance(indi, int): return indi,
                    else: return tuple(indi)
                else:
                    indi = random.choice(list(set(self.global_cm_list)-set(list(fitness_dict.keys()))))
                    return tuple(indi)
            except:
                return None
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'uniform':
                if random.random() < self.mutation_prob:
                    while True:
                        # modify the gaussian mean and standard deviation according to individual's rank
                        add = random.gauss(self.mutation_params[i][0], parent_fit_param*self.mutation_params[i][1]) + indi[i]
                        # check validity of new value in the user-defined range
                        if self.bit_limits[i][0] <= add <= self.bit_limits[i][1]: break
                    indi[i] = add
            elif self.chromosome_type[i] == 'int':
                if random.random() < self.mutation_prob:
                    indi[i] = random.randint(self.bit_limits[i][0],
                                            self.bit_limits[i][1])
            elif self.chromosome_type[i] == 'choice':
                if random.random() < self.mutation_prob:
                    indi[i] = random.choice(list(set(self.bit_limits[i]) - set([indi[i]])))
        if tuple(indi) in fitness_dict.keys(): indi = self.custom_mutate(tuple(indi), fitness_dict)
        return tuple(indi)

    def _resolve_n_jobs(self, n_tasks):
        if n_tasks <= 1:
            return 1
        if self.n_jobs == 0:
            raise ValueError("n_jobs=0 is not valid. Use 1 for sequential execution or -1 for all CPUs.")

        if self.n_jobs is None:
            jobs = 1
        elif self.n_jobs < 0:
            jobs = mp.cpu_count() + 1 + self.n_jobs
        else:
            jobs = self.n_jobs

        jobs = max(1, jobs)
        return min(jobs, n_tasks)

    def _fit_eval(self, invalid_ind, fitness_dict):
        if not invalid_ind:
            return fitness_dict

        invalid_ind = [i for i in invalid_ind if i not in fitness_dict.keys()]
        if not invalid_ind:
            return fitness_dict

        n_workers = self._resolve_n_jobs(len(invalid_ind))

        if n_workers > 1:
            # On Windows/Jupyter, notebook-defined callables under spawn can hang before any generation starts.
            start_method = mp.get_start_method(allow_none=True)
            if start_method is None:
                try:
                    start_method = mp.get_context().get_start_method()
                except Exception:
                    start_method = None
            eval_module = getattr(self.evaluate, "__module__", "")
            if os.name == "nt" and start_method == "spawn" and eval_module == "__main__":
                print("Multiprocessing disabled for evaluate defined in __main__ under spawn (notebook/interactive context). Falling back to sequential evaluation.")
                fitnesses = list(map(self.evaluate, invalid_ind))
            else:
                try:
                    pickle.dumps(self.evaluate)
                    pickle.dumps(invalid_ind)
                    payload = [(self.evaluate, ind) for ind in invalid_ind]
                    with mp.Pool(processes=n_workers) as pool:
                        fitnesses = pool.map(_evaluate_individual_mp, payload)
                except (pickle.PicklingError, AttributeError, RuntimeError, OSError) as exc:
                    print("Multiprocessing fitness evaluation unavailable (%s). Falling back to sequential evaluation." % exc)
                    fitnesses = list(map(self.evaluate, invalid_ind))
        else:
            fitnesses = list(map(self.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            fitness_dict[tuple(ind)] = fit
        return fitness_dict

    def search(self, n_generations=20, early_stopping=10, init_ratio = 0.35, crossover_ratio = 0.35):
        """
        Parameters
        ----------
        n_generations : integer, optional (default = 20)
                An integer for the number of generations to evolve the population for.

        early_stopping : int, optional (default=10)
                Integer specifying the maximum number of generations for which the algorithm can select the same best individual, after which 
                the search terminates.

        init_ratio : float, optional (default = 0.4)
            Fraction of initial population to select for next generation. Required only for algorithm 3.

        crossover_ratio : float, optional (default = 0.3)
            Fraction of crossover population to select for next generation. Required only for algorithm 3.

        
        Attributes
        ----------
        population : list,
            list of individuals from the final generation

        fitness_dict : dict,
            dictionary of all individuals evaluated by the algorithm


        Returns
        -------
        best_ind_df :  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind :  dict,
            The best individual after the last generation.

        """
        if init_ratio >=1 or crossover_ratio >=1 or (init_ratio+crossover_ratio)>=1: raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
        if self.population is not None:
            pop = self.population
            fitness_dict = self.fitness_dict
        else:
            pop = self.pop_generator(n=self.pop_size)       # list of tuples
            fitness_dict = {}
        
        # Evaluate the initial population
        fitness_dict = self._fit_eval(pop, fitness_dict)
        self.fitness_dict = fitness_dict
        self._validate_fitness_dimensions()

        best_indi_per_gen, best_indi_fitness_values, timer, total_pop, convergence, flag = [], [], [], [], 0, False
        
        from tqdm.auto import tqdm
        pbar = tqdm(range(n_generations), desc="Generation", unit="gen", position=0, leave=True)
        
        for c_gen in pbar:
            if convergence >= early_stopping:
                print("The search converged with convergence criteria = ", early_stopping)
                break
            else:
                st_time = time.time()
                cross_pop, mutant_pop, co_pop, psum = [], [], [], len(list(fitness_dict.items()))
                
                # Generate crossover population
                co_pop = self.select(pop, fitness_dict, int(math.ceil(self.crossover_size)))
                co_pop = list(itertools.combinations(list(set(co_pop)), 2))
                combi = list(itertools.combinations(list(set(pop + total_pop)), 2))
                co_pop += combi
                for child1, child2 in co_pop:
                    if (len(list(fitness_dict.items())) - psum) >= int(math.ceil(self.crossover_size)): break
                    if self.crossover_type == "SinglePoint":
                        c1, c2 = self.SinglePointCrossover(child1, child2)
                    elif self.crossover_type == "DoublePoint":
                        c1, c2 = self.DoublePointCrossover(child1, child2)
                    elif self.crossover_type == "Blend":
                        c1, c2 = self.blend(child1, child2, fitness_dict)
                    elif self.crossover_type == "Fused":
                        c1, c2 = self.fused(child1, child2, fitness_dict)
                    elif self.crossover_type == "Uniform":
                        c1, c2 = self.UniformCrossover(child1, child2)
                    if c1 in fitness_dict.keys() or c2 in fitness_dict.keys() or c1==c2: continue
                    fitness_dict = self._fit_eval([c1, c2], fitness_dict)
                    cross_pop.extend([c1, c2])
                    
                # Generate mutation population
                if self.algo == 4:
                    mu_pop = self.select(cross_pop, fitness_dict, int(math.ceil(self.mutation_size)))
                else:
                    mu_pop = self.select(pop, fitness_dict, int(math.ceil(self.mutation_size)))
                
                for mutant in mu_pop:
                    a = self.custom_mutate(mutant, fitness_dict)
                    if a is not None:
                        mutant_pop.append(a)
                        fitness_dict = self._fit_eval([a], fitness_dict)
                    else: 
                        print("All combinations exhausted. Stopping genetic algorithm iterations.")
                        flag = True
                        break
                
                # Select the next generation individuals
                total_pop = pop + cross_pop + mutant_pop
                if self.algo == 2 and c_gen != n_generations - 1:
                    pop = self.select(total_pop, fitness_dict, self.pop_size)
                elif self.algo == 3:
                    p1 = self.select(pop, fitness_dict, int(init_ratio*self.pop_size), choice="best")
                    p2 = self.select(cross_pop, fitness_dict, int(crossover_ratio*self.pop_size), choice="best")
                    p3 = self.select(mutant_pop, fitness_dict, self.crossover_size+self.mutation_size-len(p1)-len(p2), choice="best")
                    pop = p1 + p2 + p3
                else: pop = self.select(total_pop, fitness_dict, self.pop_size, choice="best")
                
                # Storing the best individuals after each generation
                best_individual = pop[0]
                if len(best_indi_per_gen)>0:
                    if best_individual==best_indi_per_gen[-1]: convergence += 1
                    else: convergence = 0
                best_indi_per_gen.append(best_individual)
                best_indi_fitness_values.append(fitness_dict[best_individual])
                tot_time = (time.time() - st_time)/(60*60)
                timer.append(tot_time)
                b1 = pd.Series(best_indi_per_gen, name='Best_individual')
                b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
                b3 = pd.Series(timer, name='Time (hours)')
                best_ind_df = pd.concat([b1, b2, b3], axis=1)
                if flag: break
        
        pbar.close()

        self.population = pop    # stores best individuals of last generation
        self.fitness_dict = fitness_dict
        best_ind_dict = {}
        for name, val in zip(self.var_names, best_individual):
            best_ind_dict[name] = val
        return best_ind_df, best_ind_dict
