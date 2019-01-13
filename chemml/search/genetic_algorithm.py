from __future__ import print_function
from builtins import range
from copy import deepcopy
from deap import base, creator, tools
import random
import pandas as pd
import time
import math


class GeneticAlgorithm(object):
    """
            A genetic algorithm class for search or optimization problems, built on top of the
            Distributed Evolutionary Algorithms in Python (DEAP) library. There are three algorithms with different genetic
            algorithm selection methods. The documentation for each method is mentioned in the documentation for the search module.

            Parameters
            ----------
            evaluate: function
                The objective function that has to be optimized. The first parameter of the objective function is a list of the trial values 
                of the hyper-parameters in the order in which they are declared in the space variable. Objective function should return a tuple.

            space: tuple, 
                A tuple of dict objects specifying the hyper-parameter space to search in. Each hyper-parameter should be a python dict object
                with the name of the hyper-parameter as the key. Value is also a dict object with one mandatory key among: 'uniform', 'int' and
                'choice' for defining floating point, integer and choice variables respectively. Values for these keys should be a list defining 
                the valid hyper-parameter search space. For uniform, a 'mutation' key is also required for which the value is 
                [mean, standard deviation] for the gaussian distribution.
                Example: 
                        ({'alpha': {'uniform': [0.00001, 1], 
                                    'mutation': [0, 1]}}, 
                        {'layers': {'int': [1, 3]}},
                        {'neurons': {'choice': range(0,200,20)}})

            weights: tuple, optional (default = (-1.0, ) )
                A tuple of integers containing fitness objective(s) for objective function(s). Ex: (1.0,) for maximizing and (-1.0,)
                for minimizing a single objective function

            pop_size: integer, optional (default = 50)
                Size of the population

            crossover_type: string, optional (default = "Blend")
                Type of crossover: OnePoint, TwoPoint, Blend 

            mutation_prob: float, optional (default = 0.4)
                Probability of mutation.

            crossover_size: float, optional (default = 0.8)
                Fraction of population to select for crossover.

            mutation_size: float, optional (default = 0.3)
                Fraction of population to select for mutation.

            algorithm: int, optional (default=1)
                The algorithm to use for the search. Algorithm descriptions are in the documentation for search method.

            initial_population: list, optional (default=None)
                The initial population for the algorithm to start with. 

            """

    def __init__(self, 
                evaluate, 
                space,
                weights=(-1.0, ), 
                pop_size=50,
                crossover_type="Blend", 
                mutation_prob=0.4,
                crossover_size=0.8,
                mutation_size=0.3,
                algorithm=3, 
                initial_population=None):

        self.Weights = weights
        self.chromosome_length = len(space)
        if self.chromosome_length <= 1:
            print("Space variable not defined. Aborting.")
            exit(code=1)
        self.chromosome_type, self.bit_limits, self.mutation_params, self.var_names = [], [], [], []
        
        for param_dict in space:
            for name in param_dict:
                self.var_names.append(name)
                var = param_dict[name]
                if 'uniform' in var:
                    self.chromosome_type.append('uniform')
                    self.bit_limits.append(var['uniform'])

                elif 'int' in var:
                    self.chromosome_type.append('int')
                    self.bit_limits.append(var['int'])

                elif 'choice' in var:
                    self.chromosome_type.append('choice')
                    self.bit_limits.append(var['choice'])
                
                if 'mutation' in var:
                    self.mutation_params.append(var['mutation'])
                else: self.mutation_params.append(None)
        
        
        self.evaluate = evaluate
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_type = crossover_type
        self.crossover_size = crossover_size
        self.mutation_size = mutation_size
        self.algo = algorithm
        self.initial_pop = initial_population
        
    def chromosome_generator(self):
        chsome = []
        if self.initial_pop is not None:
            for i in self.initial_pop:
                chsome.append(i)
        else:
            for i, j in zip(self.bit_limits, self.chromosome_type):
                if j == 'uniform':
                    chsome.append(random.uniform(i[0], i[1]))
                elif j == 'int':
                    chsome.append(random.randint(i[0], i[1]))
                elif j == 'choice':
                    chsome.append(random.choice(i))
        return chsome

    def initialize(self):
        """
        Setting up the DEAP - genetic algorithm parameters and functions.

        Notes
        -----
        Must call this function before calling any algorithm.
        """
        from deap import creator
        creator.create("FitnessMin", base.Fitness, weights=self.Weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.chromosome_generator)
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)

        def blend(ind1, ind2):
            for i in range(self.chromosome_length):
                if self.chromosome_type[i] == 'int' or self.chromosome_type[i] == 'uniform':
                    ind1[i], ind2[i] = 0.3*(ind1[i]+ind2[i]), (1-0.3)*(ind1[i]+ind2[i])
                    if self.chromosome_type[i] == 'int':
                        ind1[i], ind2[i] = int(ind1[i]), int(ind2[i])
                if self.chromosome_type[i] == 'choice':
                    ind1[i], ind2[i] = ind2[i], ind1[i]
                
        if self.crossover_type == "OnePoint":
            self.toolbox.register("mate", tools.cxOnePoint)
        elif self.crossover_type == "TwoPoint":
            self.toolbox.register("mate", tools.cxTwoPoint)
        elif self.crossover_type == "Blend":
            self.toolbox.register("mate", blend)

        # self.toolbox.register("selectTournament", tools.selTournament, tournsize=30)
        self.toolbox.register("selectRoulette", tools.selRoulette)

        def feasibility(indi):
            for x, i in zip(indi, range(len(self.bit_limits))):
                if self.chromosome_type[i] == 'uniform':
                    if not self.bit_limits[i][0] <= x <= self.bit_limits[i][1]:
                        return False
            return True

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.decorate(
            "evaluate",
            tools.DeltaPenalty(feasibility, -1000.0 * self.Weights[0]))

    def custom_mutate(self, indi):
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'uniform':
                if random.random() < self.mutation_prob:
                    indi[i] += random.gauss(self.mutation_params[i][0],
                                            self.mutation_params[i][1])
            elif self.chromosome_type[i] == 'int':
                if random.random() < self.mutation_prob:
                    indi[i] = random.randint(self.bit_limits[i][0],
                                            self.bit_limits[i][1])
            elif self.chromosome_type[i] == 'choice':
                if random.random() < self.mutation_prob:
                    indi[i] = random.choice(self.bit_limits[i])

    def search(self, n_generations=20, early_stopping=10, init_ratio = 0.35, crossover_ratio = 0.35):
        """
        Algorithm 1:
            Initial population is instantiated. 
            Roulette wheel selection is used for selecting individuals for crossover and mutation.
            The initial population, crossovered and mutated individuals form the pool of individuals from which the best
            n members are selected as the initial population for the next generation, where n is the size of population.

        Algorithm 2:
            Initial population is instantiated.
            Roulette wheel selection is used for selecting individuals for crossover and mutation.
            The initial population, crossovered and mutated individuals form 3 different pools of individuals. Based on
            input parameters 1 and 2, members are selected from each of these pools to form the initial population for the
            next generation. Fraction of mutated members to select for next generation is decided based on the two input
            parameters and the size of initial population.

        Algorithm 3:
            Initial population is instantiated.
            Roulette wheel selection is used for selecting individuals for crossover and mutation.
            The initial population, crossovered and mutated individuals form the pool of individuals from which n members
            are selected using Roulette wheel selection, but without replacement to ensure uniqueness of members in the next
            generation, as the initial population for the next generation, where n is the size of population.


        Parameters
        ----------
        n_generations: integer, optional (default = 20)
                An integer for the number of generations for evolving the population

        early_stopping: int, optional (default=10)
                Integer specifying the maximum number of generations for which the algorithm can select the same best individual, after which 
                the search terminates.

        init_ratio: float, optional (default = 0.4)
            Fraction of initial population to select for next generation. Required only for algorithm 2.

        crossover_ratio: float, optional (default = 0.3)
            Fraction of crossover population to select for next generation. Required only for algorithm 2.

        
        Returns
        -------
        best_ind_df:  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind:  list,
            The best individual after the last generation.

        total_pop: list,
            List of individuals in the final population 

        """

        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the initial population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Generate and evaluate crossover and mutation population
        if self.algo == 3:
            co_pop = []
            while len(co_pop) < int(math.ceil(self.crossover_size*len(pop))):
                c = self.toolbox.selectRoulette(pop, 1)
                if c not in co_pop:
                    co_pop = co_pop + c
        else:
            co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(self.crossover_size*len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(self.mutation_size*len(pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))

        for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
            c1, c2 = deepcopy(child1), deepcopy(child2)
            self.toolbox.mate(child1, child2)
            if self.crossover_type == 'Blend':
                for i in range(self.chromosome_length):
                    if self.chromosome_type[i] == 'int':
                        child1[i], child2[i] = int(child1[i]), int(child2[i])
                    if self.chromosome_type[i] == 'choice' and type(child1[i])==str:
                        child1[i], child2[i] = c1[i], c2[i]
            del child1.fitness.values
            del child2.fitness.values

        for mutant in mu_pop:
            self.custom_mutate(mutant)
            del mutant.fitness.values

        # Evaluate the crossover and mutated population
        if self.algo == 2:
            a, b = int(math.ceil(init_ratio*len(pop))), int(math.ceil(crossover_ratio*len(pop)))
            total_pop = tools.selBest(pop, a) + tools.selBest(co_pop, b) + tools.selBest(mu_pop, len(pop)-a-b)
        else:
            total_pop = pop + co_pop + mu_pop
        invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        timer = []
        convergence = 0
        for _ in range(n_generations):
            if convergence >= early_stopping:
                print("The search converged with convergence criteria = ", early_stopping)
                break
            else:
                st_time = time.time()
                # Select the next generation individuals
                offspring = tools.selBest(total_pop, self.pop_size)
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring))
                if self.algo == 3:
                    co_pop = []
                    while len(co_pop) < int(math.ceil(self.crossover_size*len(pop))):
                        c = self.toolbox.selectRoulette(pop, 1)
                        if c not in co_pop:
                            co_pop = co_pop + c
                else:
                    co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(self.crossover_size*len(pop))))
                co_pop = list(map(self.toolbox.clone, co_pop))
                mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(self.mutation_size*len(pop))))
                mu_pop = list(map(self.toolbox.clone, mu_pop))

                for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
                    c1, c2 = deepcopy(child1), deepcopy(child2)
                    self.toolbox.mate(child1, child2)
                    if self.crossover_type == 'Blend':
                        for i in range(self.chromosome_length):
                            if self.chromosome_type[i] == 'int':
                                child1[i], child2[i] = int(child1[i]), int(child2[i])
                            if self.chromosome_type[i] == 'choice' and type(child1[i])==str:
                                child1[i], child2[i] = c1[i], c2[i]
                    del child1.fitness.values
                    del child2.fitness.values

                for mutant in mu_pop:
                    self.custom_mutate(mutant)
                    del mutant.fitness.values

                # Evaluate the crossover and mutated population
                if self.algo == 2:
                    a, b = int(math.ceil(init_ratio*len(pop))), int(math.ceil(crossover_ratio*len(pop)))
                    total_pop = tools.selBest(pop, a) + tools.selBest(co_pop, b) + tools.selBest(mu_pop, len(pop)-a-b)
                else:
                    total_pop = offspring + co_pop + mu_pop
                invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit


                # Storing the best individuals after each generation
                best_individual = tools.selBest(total_pop, 1)[0]
                if len(best_indi_per_gen)>0 and best_individual==best_indi_per_gen[-1]:
                    convergence += 1
                best_indi_per_gen.append(list(best_individual))
                best_indi_fitness_values.append(best_individual.fitness.values[0])

                tot_time = (time.time() - st_time)/(60*60)
                timer.append(tot_time)

                b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
                b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
                b3 = pd.Series(timer, name='Time')
                best_ind_df = pd.concat([b1, b2, b3], axis=1)

        print("\n \n Best Individuals of each generation are:  \n \n" , best_ind_df)
        print("\n \n Best individual after %s evolutions is %s " % (n_generations, best_individual))
        del creator.FitnessMin
        del creator.Individual
        best_ind_dict = {}
        for name, val in zip(self.var_names, best_individual):
            best_ind_dict[name] = val
        return best_ind_df, best_ind_dict, total_pop

