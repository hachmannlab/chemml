from __future__ import print_function
from builtins import range
import random
import pandas as pd
import time
import math
import numpy as np
from copy import deepcopy
import itertools

class GeneticAlgorithm(object):
    """
            A python implementation of real-valued, genetic algorithm for solving optimization problems.

            Parameters
            ----------
            evaluate: function
                The objective function that has to be optimized. The first parameter of the objective function is a list of the trial values 
                of the hyper-parameters in the order in which they are declared in the space variable.

            space: tuple, 
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

            fitness: str, optional (default = 'Max')
                Maximize (Max) or minimize (Min) the objective function.
                
            pop_size: integer, optional (default = 50)
                Size of the population

            crossover_type: string, optional (default = "Blend")
                Type of crossover: SinglePoint, DoublePoint, Blend 

            mutation_prob: float, optional (default = 0.4)
                Probability of mutation.

            crossover_size: float, optional (default = 0.8)
                Fraction of population to select for crossover.

            mutation_size: float, optional (default = 0.3)
                Fraction of population to select for mutation.

            algorithm: int, optional (default=1)
                The algorithm to use for the search. Look at the 'search' method for a description of the various algorithms.

            initial_population: list, optional (default=None)
                The initial population for the algorithm to start with. If not provided, initial population is randomly generated.

            """

    def __init__(self, 
                evaluate, 
                space,
                fitness="Max", 
                pop_size=50,
                crossover_type="Blend", 
                mutation_prob=0.6,
                crossover_size=0.5,
                mutation_size=0.3,
                algorithm=1,
                initial_population=None):

        self.chromosome_length = len(space)
        if self.chromosome_length <=2 and crossover_type == "DoublePoint": raise Exception('Double point crossover not possible for gene length 2.')
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
        if fitness.lower() == 'max': self.fit_val = 1
        else: self.fit_val = -1
        self.population, self.fitness_dict = None, {}
        
    def pop_generator(self, n):
        pop = []
        if self.initial_pop is not None:
            for i in self.initial_pop:
                pop.append(tuple(i))
        else:
            for _ in range(n):
                pop.append(self.chromosome_generator())
        return pop

    def chromosome_generator(self):
        chsome = []
        for i, j in zip(self.bit_limits, self.chromosome_type):
            if j == 'uniform':
                chsome.append(random.uniform(i[0], i[1]))
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

    def blend(self, ind1, ind2, z=0.4):
        ind1, ind2 = list(ind1), list(ind2)
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'choice':
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:
                chi = (1 + 2*z) * random.random() - z
                try:
                    ind1[i], ind2[i] = (1-chi)*ind1[i]+chi*ind2[i], chi*ind2[i]+(1-chi)*ind1[i]
                except: print(chi, ind1, ind2)
                if self.chromosome_type[i] == 'int':
                    ind1[i], ind2[i] = int(ind1[i]), int(ind2[i])
        return tuple(deepcopy(ind1)), tuple(deepcopy(ind2))

    def RouletteWheelSelection(self, population, fit_dict, num):
        o_fits = [fit_dict[i] for i in population]
        fitnesses = [(((i-min(o_fits))/(max(o_fits)-min(o_fits))) + 1) for i in o_fits]
        if self.fit_val == -1:
            fitnesses = [fit**self.fit_val for fit in fitnesses]
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

    def selectbest(self, pop, n, fitness_dict):
        best = []
        o_fits = [fitness_dict[i] for i in pop]
        fitnesses = [(((i-min(o_fits))/(max(o_fits)-min(o_fits))) + 1) for i in o_fits]
        if self.fit_val == -1:
            fitnesses = [fit**self.fit_val for fit in fitnesses]
        fits_sort = sorted(fitnesses, reverse=True)
        for i in range(min(n, len(pop))):
            best.append(deepcopy(pop[fitnesses.index(fits_sort[i])]))
        return best

    def custom_mutate(self, indi, fitness_dict):
        indi = list(indi)
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'uniform':
                if random.random() < self.mutation_prob:
                    while True:
                        add = random.gauss(self.mutation_params[i][0], self.mutation_params[i][1]) + indi[i]
                        if self.bit_limits[i][0] <= add <= self.bit_limits[i][1]: break
                    indi[i] = add
            elif self.chromosome_type[i] == 'int':
                if random.random() < self.mutation_prob:
                    indi[i] = random.randint(self.bit_limits[i][0],
                                            self.bit_limits[i][1])
            elif self.chromosome_type[i] == 'choice':
                if random.random() < self.mutation_prob:
                    indi[i] = random.choice(self.bit_limits[i])
        if tuple(indi) in fitness_dict.keys(): indi = self.custom_mutate(indi, fitness_dict)
        return tuple(indi)

    def search(self, n_generations=20, early_stopping=10, init_ratio = 0.35, crossover_ratio = 0.35):
        """
        Algorithm 1:
            Initial population is instantiated. 
            Roulette wheel selection is used for selecting individuals for crossover and mutation.
            The initial population, crossovered and mutated individuals form the pool of individuals from which the best
            n members are selected as the initial population for the next generation, where n is the size of population.

        Algorithm 2:
            Same as algorithm 1 but when selecting individuals for next generation, n members are selected using Roulette wheel selection.

        Algorithm 3:
            Same as algorithm 1 but when selecting individuals for next generation, best members from each of the three pools (initital population, crossover and mutation) are selected according to the input parameters in the search method.

        Algorithm 4:
            Same as algorithm 1 but mutation population is selected from the crossover population and not from the parents directly.


        Parameters
        ----------
        n_generations: integer, optional (default = 20)
                An integer for the number of generations to evolve the population for.

        early_stopping: int, optional (default=10)
                Integer specifying the maximum number of generations for which the algorithm can select the same best individual, after which 
                the search terminates.

        init_ratio: float, optional (default = 0.35)
            Fraction of initial population to select for next generation. Required only for algorithm 3.

        crossover_ratio: float, optional (default = 0.35)
            Fraction of crossover population to select for next generation. Required only for algorithm 3.

        
        Attributes
        ----------
        population: list,
            list of individuals from the final generation

        fitness_dict: dict,
            dictionary of all individuals evaluated by the algorithm


        Returns
        -------
        best_ind_df:  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind:  dict,
            The best individual after the last generation.

        """
        def fit_eval(invalid_ind, fitness_dict):
            if invalid_ind: 
                invalid_ind = [i for i in invalid_ind if i not in fitness_dict.keys()]
                fitnesses = list(map(self.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    fitness_dict[tuple(ind)] = fit
            return fitness_dict


        if init_ratio >=1 or crossover_ratio >=1 or (init_ratio+crossover_ratio)>=1: raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
        if self.population is not None:
            # pop = [i for i in self.population]
            # fitness_dict = self.population
            pop = self.population
            fitness_dict = self.fitness_dict
        else:
            pop = self.pop_generator(n=self.pop_size)       # list of tuples
            fitness_dict = {}
        
        # Evaluate the initial population
        fitness_dict = fit_eval(pop, fitness_dict)

        best_indi_per_gen, best_indi_fitness_values, timer, total_pop, convergence = [], [], [], [], 0
        for _ in range(n_generations):
            if convergence >= early_stopping:
                print("The search converged with convergence criteria = ", early_stopping)
                break
            else:
                st_time = time.time()
                cross_pop, mutant_pop, co_pop, psum = [], [], [], len(list(fitness_dict.items()))
                # Generate crossover population
                co_pop = self.RouletteWheelSelection(pop, fitness_dict, int(math.ceil(self.crossover_size*len(pop))))
                co_pop = list(itertools.combinations(list(set(co_pop)), 2))
                combi = list(itertools.combinations(list(set(pop + total_pop)), 2))
                for child1, child2 in co_pop + combi:
                    if (len(list(fitness_dict.items())) - psum) >= int(math.ceil(self.crossover_size*len(pop))): break
                    if self.crossover_type == "SinglePoint":
                        c1, c2 = self.SinglePointCrossover(child1, child2)
                    elif self.crossover_type == "DoublePoint":
                        c1, c2 = self.DoublePointCrossover(child1, child2)
                    elif self.crossover_type == "Blend":
                        c1, c2 = self.blend(child1, child2)
                    if c1 in fitness_dict.keys() or c2 in fitness_dict.keys() or c1==c2: continue
                    fitness_dict = fit_eval([c1, c2], fitness_dict)
                    cross_pop.extend([c1, c2])
                    
                # Generate mutation population
                if self.algo == 4:
                    mu_pop = self.RouletteWheelSelection(cross_pop, fitness_dict, int(math.ceil(self.pop_size*self.mutation_size)))
                else:
                    mu_pop = self.RouletteWheelSelection(pop, fitness_dict, int(math.ceil(self.mutation_size*len(pop))))
                
                for mutant in mu_pop:
                    mutant_pop.append(self.custom_mutate(mutant, fitness_dict))
                fitness_dict = fit_eval(mutant_pop, fitness_dict)
                
                # Select the next generation individuals
                total_pop = pop + cross_pop + mutant_pop
                if self.algo == 2:
                    pop = self.RouletteWheelSelection(total_pop, fitness_dict, self.pop_size)
                elif self.algo == 3:
                    p1 = self.selectbest(pop, int(init_ratio*self.pop_size), fitness_dict)
                    p2 = self.selectbest(cross_pop, int(crossover_ratio*self.pop_size), fitness_dict)
                    p3 = self.selectbest(mutant_pop, self.pop_size-len(p1)-len(p2), fitness_dict)
                    pop = p1 + p2 + p3
                else: pop = self.selectbest(total_pop, self.pop_size, fitness_dict)
                
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
    

        self.population = pop    # stores best individuals of last generation
        self.fitness_dict = fitness_dict
        best_ind_dict = {}
        for name, val in zip(self.var_names, best_individual):
            best_ind_dict[name] = val
        return best_ind_df, best_ind_dict
