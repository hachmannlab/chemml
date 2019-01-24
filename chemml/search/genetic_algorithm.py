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
                the valid hyper-parameter search space (lower and upper bounds for int and uniform). For uniform, a 'mutation' key is also required for which the value is 
                [mean, standard deviation] for the gaussian distribution.
                Example: 
                        ({'alpha': {'uniform': [0.00001, 1], 
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
                The algorithm to use for the search. Algorithm descriptions are in the documentation for search method.

            mutation_ratio: int, optional (default=None)
                Fraction of crossover population to select for mutation. Required for algorithm 4 only.

            initial_population: list, optional (default=None)
                The initial population for the algorithm to start with. If not provided, initial population is randomly generated.

            """

    def __init__(self, 
                evaluate, 
                space,
                fitness="Max", 
                pop_size=50,
                crossover_type="Blend", 
                mutation_prob=0.4,
                crossover_size=0.5,
                mutation_size=0.3,
                algorithm=1,
                mutation_ratio = None,
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
        if fitness.lower() == 'max':
            self.fit_val = 1
        else: self.fit_val = -1
        self.population, self.ex_fitness_dict = None, {}
        if self.algo == 4 and mutation_ratio is None: raise Exception("Mutation parameter for algorithm 4 not provided.")
        self.mu_ratio = mutation_ratio
        
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
        return tuple(y1), tuple(y2)

    def DoublePointCrossover(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        nVar = len(x1)
        cc = random.sample(range(1,nVar), 2)   
        c1 = min(cc)
        c2 = max(cc)
        y1 = x1[0:c1]+x2[c1:c2]+x1[c2:nVar]				
        y2 = x2[0:c1]+x1[c1:c2]+x2[c2:nVar]      
        return tuple(y1), tuple(y2)

    def blend(self, ind1, ind2):
        ind1, ind2 = list(ind1), list(ind2)
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'choice':
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:
                ind1[i], ind2[i] = 0.3*ind1[i]+0.7*ind2[i], 0.3*ind2[i]+0.7*ind1[i]
                if self.chromosome_type[i] == 'int':
                    ind1[i], ind2[i] = int(ind1[i]), int(ind2[i])
        return tuple(ind1), tuple(ind2)

    def RouletteWheelSelection(self, fit_dict, num):
        population = fit_dict.keys()                                            # list of tuples
        fitnesses = [fit_dict[i]['w_fit'] for i in population]
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

    def TournamentSelection(self, C, m):
        nVar=len(C)
        c=[]
        S = random.sample(range(0,nVar), m)
        for i in range (0,m):
            c.append(C[S[i]])
        c, S = (list(t)for t in zip(*sorted(zip(c, S))))
        return(S[0])

    def selectbest(self, pop, n, fitness_dict):
        best = []
        fits = [fitness_dict[i]['w_fit'] for i in pop]
        fits_sort = sorted(fits, reverse=True)
        for i in range(min(n, len(pop))):
            best.append(pop[fits.index(fits_sort[i])])
        return best

    def custom_mutate(self, indi):
        indi = list(indi)
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 'uniform':
                if random.random() < self.mutation_prob:
                    add = self.bit_limits[i][0] -1
                    while self.bit_limits[i][0] <= add <= self.bit_limits[i][1]:
                        add = random.gauss(self.mutation_params[i][0], self.mutation_params[i][1]) + indi[i]
                    indi[i] += add
            elif self.chromosome_type[i] == 'int':
                if random.random() < self.mutation_prob:
                    indi[i] = random.randint(self.bit_limits[i][0],
                                            self.bit_limits[i][1])
            elif self.chromosome_type[i] == 'choice':
                if random.random() < self.mutation_prob:
                    indi[i] = random.choice(self.bit_limits[i])
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
                An integer for the number of generations for evolving the population

        early_stopping: int, optional (default=10)
                Integer specifying the maximum number of generations for which the algorithm can select the same best individual, after which 
                the search terminates.

        init_ratio: float, optional (default = 0.4)
            Fraction of initial population to select for next generation. Required only for algorithm 3.

        crossover_ratio: float, optional (default = 0.3)
            Fraction of crossover population to select for next generation. Required only for algorithm 3.

        
        Returns
        -------
        best_ind_df:  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind:  dict,
            The best individual after the last generation.

        """
        def fit_eval(invalid_ind):
            fitnesses = list(map(self.evaluate, invalid_ind))
            wt_fits = [(((i-min(fitnesses))/(max(fitnesses)-min(fitnesses))) + 1) for i in fitnesses]
            if self.fit_val == -1:
                weighted_fitnesses = [fit**self.fit_val for fit in wt_fits]
            else: weighted_fitnesses = wt_fits
            for ind, fit, wfit in zip(invalid_ind, fitnesses, weighted_fitnesses):
                temp = {'fit': fit, 'w_fit': wfit}
                fitness_dict[ind] = temp


        if init_ratio >=1 or crossover_ratio >=1 or (init_ratio+crossover_ratio)>=1: raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
        if self.population is not None:
            pop = [i for i in self.population]
            fitness_dict = self.population
        else:
            pop = self.pop_generator(n=self.pop_size)       # list of tuples
            fitness_dict = {}
        
        # Evaluate the initial population
        invalid_ind = [ind for ind in pop if ind not in fitness_dict]
        if invalid_ind: fit_eval(invalid_ind)

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
                cross_pop, mutant_pop = [], []
                # Generate crossover population
                c_total = self.RouletteWheelSelection(fitness_dict, math.ceil(self.crossover_size*len(pop)))
                combi = list(itertools.combinations(list(set(c_total)), 2))
                co_pop = []
                for i, j in zip(list(set(c_total))[::2], list(set(c_total))[1::2]):
                    co_pop.append((i, j))
                c_rem = [i for i in combi if i not in co_pop]
                diff = math.ceil(self.crossover_size*len(pop)/2) - len(co_pop)
                if diff>0: co_pop += c_rem[:diff]
                for child1, child2 in co_pop:
                    if self.crossover_type == "SinglePoint":
                        c1, c2 = self.SinglePointCrossover(child1, child2)
                    elif self.crossover_type == "DoublePoint":
                        c1, c2 = self.DoublePointCrossover(child1, child2)
                    elif self.crossover_type == "Blend":
                        c1, c2 = self.blend(child1, child2)
                    cross_pop.extend([deepcopy(c1), deepcopy(c2)])
                fit_eval(cross_pop)

                # Generate mutation population
                if self.algo == 4:
                    mu_pop = self.RouletteWheelSelection({ind:fitness_dict[ind] for ind in cross_pop}, math.ceil(self.pop_size*self.mu_ratio))
                else:
                    mu_pop = self.RouletteWheelSelection(fitness_dict, math.ceil(self.mutation_size*len(pop)))
                
                for mutant in mu_pop:
                    m = self.custom_mutate(mutant)
                    mutant_pop.append(m)
                fit_eval(mutant_pop)
                
                # Select the next generation individuals
                total_pop = pop + cross_pop + mutant_pop
                if self.algo == 2:
                    pop = self.RouletteWheelSelection(fitness_dict, self.pop_size)
                elif self.algo == 3:
                    p1 = self.selectbest(pop, int(init_ratio*self.pop_size), fitness_dict)
                    p2 = self.selectbest(cross_pop, int(crossover_ratio*self.pop_size), fitness_dict)
                    p3 = self.selectbest(mutant_pop, self.pop_size-len(p1)-len(p2), fitness_dict)
                    pop = p1 + p2 + p3
                else: pop = self.selectbest(total_pop, self.pop_size, fitness_dict)
                for i in fitness_dict:
                    self.ex_fitness_dict[i] = fitness_dict[i]
                fitness_dict = {ind:fitness_dict[ind] for ind in pop}
                
                # Storing the best individuals after each generation
                best_individual = pop[0]
                if len(best_indi_per_gen)>0:
                    if best_individual==best_indi_per_gen[-1]:
                        convergence += 1
                    else: convergence = 0
                best_indi_per_gen.append(best_individual)
                best_indi_fitness_values.append(fitness_dict[best_individual]['fit'])
                tot_time = (time.time() - st_time)/(60*60)
                timer.append(tot_time)
                b1 = pd.Series(best_indi_per_gen, name='Best_individual')
                b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
                b3 = pd.Series(timer, name='Time (hours)')
                best_ind_df = pd.concat([b1, b2, b3], axis=1)
    

        self.population = fitness_dict    # stores best individuals of last generation
        best_ind_dict = {}
        for name, val in zip(self.var_names, best_individual):
            best_ind_dict[name] = val
        return best_ind_df, best_ind_dict
