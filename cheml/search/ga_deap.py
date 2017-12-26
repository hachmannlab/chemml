from deap import base, creator, tools
import random
import pandas as pd
import time
import math

start_time = time.time()

class GA_DEAP(object):
    def __init__(self, Evaluate, Weights = (1,), chromosome_length = 1, chromosome_type = (1,), bit_limits = ((0,10),), \
                 pop_size = 50, crossover_prob = 0.4, crossover_type = "Blend", mutation_prob = 0.4, \
                 mut_float_param_1 = 0, mut_float_param_2 = 1, mut_int_param_1 = 1, mut_int_param_2 = 10, n_generations = 100):

        """
        Evaluate: The objective function that has to be optimized. The 1st parameter of the objective function should be
        an object of type <chromosome_type>

        Weights: A tuple containing fitness objective(s) for objective function(s). Ex: (1.0,) for maximizing and (-1.0,)
        for minimizing a single objective function

        chromosome_length: An integer which specifies the length of chromosome/individual

        chromosome_type: A tuple of length chromosome_length describing the type of each bit of the chromosome.
        0 for floating type and 1 for integer type. Keep all integer types first followed by the floating types.

        bit_limits: A tuple of tuples containing the lower and upper limits for each bit

        pop_size: An integer which denotes the size of the population (keep this > 50 always)

        crossover_prob: A floating point number that denotes the Probability of crossover

        crossover_type: A string denoting the type of crossover: Options are: OnePoint, TwoPoint, Blend or Uniform

        mutation_prob: A floating point number that denotes the probability of mutation.

        mut_float_param_1: Value of the mean of the Gaussian distribution for the Gaussian type mutation

        mut_float_param_2: Value of the standard deviation of the Gaussian distribution for the Gaussian type mutation

        mut_int_param_1: Value of the lower limit (inclusive) of the integer which is to be used for integer type mutation

        mut_int_param_2: Value of the upper limit (inclusive) of the integer which is to be used for integer type mutation

        n_generations: An integer for the number of generations for evolving the population
        """

        self.Weights = Weights
        self.chromosome_length = chromosome_length
        self.chromosome_type = chromosome_type
        self.bit_limits = bit_limits
        self.evaluate = Evaluate
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_type = crossover_type
        self.mut_float_param_1 = mut_float_param_1
        self.mut_float_param_2 = mut_float_param_2
        self.mut_int_param_1 = mut_int_param_1
        self.mut_int_param_2 = mut_int_param_2
        self.n_generations = n_generations
        self.n_integers = 0
        for i in chromosome_type:
            if i == 1:
                self.n_integers += 1

    def chromosome_generator(self):
        A = []
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 0:
                A.append(random.uniform(self.bit_limits[i][0], self.bit_limits[i][1]))
            else:
                A.append(random.randint(self.bit_limits[i][0], self.bit_limits[i][1]))
        return A

    def fit(self):
        creator.create("FitnessMin", base.Fitness, weights=self.Weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.chromosome_generator)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        if self.crossover_type == "OnePoint":
            self.toolbox.register("mate", tools.cxOnePoint)
        elif self.crossover_type == "TwoPoint":
            self.toolbox.register("mate", tools.cxTwoPoint)
        elif self.crossover_type == "Uniform":
            self.toolbox.register("mate", tools.cxUniform, indpb=self.crossover_prob)
        elif self.crossover_type == "Blend":
            self.toolbox.register("mate", tools.cxBlend, alpha = 0.5)

        self.toolbox.register("selectTournament", tools.selTournament, tournsize=30)
        self.toolbox.register("selectRoulette", tools.selRoulette)

        def feasibility(indi):
            for x,i in zip(indi,range(self.chromosome_length)):
                if self.bit_limits[i][0] <= x <= self.bit_limits[i][1]:
                    continue
                else:
                    return False
            return True

        def distance(indi):
            s = 0
            for x,i in zip(indi, range(self.chromosome_length)):
                low = self.bit_limits[i][0]
                up = self.bit_limits[i][1]
                if x < low:
                    s += ((x-low)*100)**2
                elif x > up:
                    s += ((x-up)*100)**2
            return s

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, -100.0*self.Weights[0], distance))

    def custom_mutate(self, indi):
        for i in range(self.chromosome_length):
            if self.chromosome_type[i] == 0:
                if random.random() < self.mutation_prob:
                    indi[i] += random.gauss(self.mut_float_param_1, self.mut_float_param_2)
            elif self.chromosome_type[i] == 1:
                if random.random() < self.mutation_prob:
                    if self.n_integers <= 1:
                        indi[i] = random.randint(self.mut_int_param_1, self.mut_int_param_2)
                    else:
                        indi[i] = random.randint(self.mut_int_param_1[i], self.mut_int_param_2[i])

    def algorithm_1(self):
        """
        Initial population is instantiated.
        Roulette wheel selection is used for selecting individuals for crossover and mutation.
        The initial population, crossovered and mutated individuals form the pool of individuals from which the best -
        n members are selected as the initial population for the next generation, where n is the size of population.

        :return:
        best_ind_df:  A pandas dataframe of best individuals of each generation
        best_ind:  The best individual of the last generation.
        """
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8*len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3*len(pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))

        for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
            self.toolbox.mate(child1, child2)
            for i in range(self.chromosome_length):
                if self.chromosome_type[i] == 1:
                    child1[i] = int(child1[i])
                    child2[i] = int(child2[i])
            del child1.fitness.values
            del child2.fitness.values

        for mutant in mu_pop:
            self.custom_mutate(mutant)
            del mutant.fitness.values

        # Evaluate the crossover and mutated population
        total_pop = pop + co_pop + mu_pop
        invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

#        fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            offspring = tools.selBest(total_pop, self.pop_size)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8*len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3*len(pop))))
            mu_pop = list(map(self.toolbox.clone, mu_pop))

            for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
                self.toolbox.mate(child1, child2)
                for i in range(self.chromosome_length):
                    if self.chromosome_type[i] == 1:
                        child1[i] = int(child1[i])
                        child2[i] = int(child2[i])
                del child1.fitness.values
                del child2.fitness.values

            for mutant in mu_pop:
                self.custom_mutate(mutant)
                del mutant.fitness.values

            # Evaluate the crossover and mutated population
            total_pop = offspring + co_pop + mu_pop
            invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            # Storing the best individuals after each generation
            best_individual = tools.selBest(total_pop, 1)[0]
            best_indi_per_gen.append(list(best_individual))
            best_indi_fitness_values.append(best_individual.fitness.values[0])


            tot_time = (time.time() - st_time)/(60*60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
            b3 = pd.Series(timer, name = 'Time')
            best_ind_df = pd.concat([b1,b2,b3], axis=1)
            best_ind_df.to_csv('best_ind.csv',index=False)


        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
        best_ind_df = pd.concat([b1,b2], axis=1)
        best_ind_df.to_csv('best_ind.csv',index=False)

        # best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n" , best_ind_df
        print "\n \n" , self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algorithm_2(self, algo_param_1 = 0.4, algo_param_2 = 0.3):
        """
        Initial population is instantiated.
        Roulette wheel selection is used for selecting individuals for crossover and mutation.
        The initial population, crossovered and mutated individuals form 3 different pools of individuals. Based on
        input parameters 1 and 2, members are selected from each of these pools to form the initial population for the
        next generation. Fraction of mutated members to select for next generation is decided based on the 2 input
        parameters and the size of initial population.

        :param:
        algo_param_1: Float, default = 0.4
        Fraction of initial population to select for next generation

        algo_param_2: Float, default = 0.3
        Fraction of crossover population to select for next generation

        :return:
        best_ind_df:  A pandas dataframe of best individuals of each generation
        best_ind:  The best individual of the last generation.
        """
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8*len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3*len(pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))

        for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
            self.toolbox.mate(child1, child2)
            for i in range(self.chromosome_length):
                if self.chromosome_type[i] == 1:
                    child1[i] = int(child1[i])
                    child2[i] = int(child2[i])
            del child1.fitness.values
            del child2.fitness.values

        for mutant in mu_pop:
            self.custom_mutate(mutant)
            del mutant.fitness.values

        # Evaluate the crossover and mutated population
        total_pop = pop + co_pop + mu_pop
        invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

#        fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            off_pop = tools.selBest(pop, int(math.ceil(algo_param_1*len(pop))))
            off_co = tools.selBest(co_pop, int(math.ceil(algo_param_2*len(pop))))
            off_mu = tools.selBest(mu_pop, len(pop)-len(off_pop)-len(off_co) )
            offspring = off_pop + off_co + off_mu
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8*len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3*len(pop))))
            mu_pop = list(map(self.toolbox.clone, mu_pop))

            for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
                self.toolbox.mate(child1, child2)
                for i in range(self.chromosome_length):
                    if self.chromosome_type[i] == 1:
                        child1[i] = int(child1[i])
                        child2[i] = int(child2[i])
                del child1.fitness.values
                del child2.fitness.values

            for mutant in mu_pop:
                self.custom_mutate(mutant)
                del mutant.fitness.values

            # Evaluate the crossover and mutated population
            total_pop = offspring + co_pop + mu_pop
            invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop = offspring

            # Storing the best individuals after each generation
            best_individual = tools.selBest(total_pop, 1)[0]
            best_indi_per_gen.append(list(best_individual))
            best_indi_fitness_values.append(best_individual.fitness.values[0])


            tot_time = (time.time() - st_time)/(60*60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
            b3 = pd.Series(timer, name = 'Time')
            best_ind_df = pd.concat([b1,b2,b3], axis=1)
            best_ind_df.to_csv('best_ind.csv',index=False)


        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])

        b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
        best_ind_df = pd.concat([b1,b2], axis=1)
        best_ind_df.to_csv('best_ind.csv',index=False)

    #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n" , best_ind_df
        print "\n \n" , self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algorithm_3(self):
        """
        Initial population is instantiated.
        Roulette wheel selection is used for selecting individuals for crossover and mutation.
        The initial population, crossovered and mutated individuals form the pool of individuals from which n members
        are selected using Roulette wheel selection, but without replacement to ensure uniqueness of members in the next
        generation, as the initial population for the next generation, where n is the size of population.

        :return:
        best_ind_df:  A pandas dataframe of best individuals of each generation
        best_ind:  The best individual of the last generation.
        """
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8*len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3*len(pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))

        for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
            self.toolbox.mate(child1, child2)
            for i in range(self.chromosome_length):
                if self.chromosome_type[i] == 1:
                    child1[i] = int(child1[i])
                    child2[i] = int(child2[i])
            del child1.fitness.values
            del child2.fitness.values

        for mutant in mu_pop:
            self.custom_mutate(mutant)
            del mutant.fitness.values

            # Evaluate the crossover and mutated population
        total_pop = pop + co_pop + mu_pop
        invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

#        fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            #offspring = tools.selBest(total_pop, self.pop_size)
            temp_pop = total_pop
            offspring = []
            for i in range(len(pop)):
                off = self.toolbox.selectRoulette(temp_pop, 1)
                offspring.append(off[0])
                temp_pop.remove(off[0])
            # Clone the selected individuals
            # offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8*len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3*len(pop))))
            mu_pop = list(map(self.toolbox.clone, mu_pop))

            for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
                self.toolbox.mate(child1, child2)
                for i in range(self.chromosome_length):
                    if self.chromosome_type[i] == 1:
                        child1[i] = int(child1[i])
                        child2[i] = int(child2[i])
                del child1.fitness.values
                del child2.fitness.values

            for mutant in mu_pop:
                self.custom_mutate(mutant)
                del mutant.fitness.values

            # Evaluate the crossover and mutated population
            total_pop = offspring + co_pop + mu_pop
            invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Storing the best individuals after each generation
            best_individual = tools.selBest(total_pop, 1)[0]
            best_indi_per_gen.append(list(best_individual))
            best_indi_fitness_values.append(best_individual.fitness.values[0])

            tot_time = (time.time() - st_time)/(60*60)
            timer.append(tot_time)
            b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
            b3 = pd.Series(timer, name = 'Time')
            best_ind_df = pd.concat([b1,b2,b3], axis=1)
            best_ind_df.to_csv('best_ind.csv',index=False)


        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
        best_ind_df = pd.concat([b1,b2], axis=1)
        best_ind_df.to_csv('best_ind.csv',index=False)

    #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n" , best_ind_df
        print "\n \n" , self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algorithm_4(self, algo_param_4):
        """
        Initial population is instantiated.
        Roulette wheel selection is used for selecting individuals for crossover. For mutation, fraction of individuals
        are selected from the crossover population based on input parameter.
        The initial population, crossovered and mutated individuals form the pool of individuals from which the best -
        n members are selected as the initial population for the next generation, where n is the size of population.

        :param:
        algo_param_4: Float, default = 0.4
        Fraction of crossover population to select for mutation

        :return:
        best_ind_df:  A pandas dataframe of best individuals of each generation
        best_ind:  The best individual of the last generation.
        """
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8*len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(co_pop, int(math.ceil(algo_param_4*len(co_pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))
        for i in mu_pop:
            if i in co_pop:
                co_pop.remove(i)

        for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
            self.toolbox.mate(child1, child2)
            for i in range(self.chromosome_length):
                if self.chromosome_type[i] == 1:
                    child1[i] = int(child1[i])
                    child2[i] = int(child2[i])
            del child1.fitness.values
            del child2.fitness.values

        for mutant in mu_pop:
            self.custom_mutate(mutant)
            del mutant.fitness.values

        # Evaluate the crossover and mutated population
        total_pop = pop + co_pop + mu_pop
        invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

#        fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
                # Select the next generation individuals
            offspring = tools.selBest(total_pop, self.pop_size)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8*len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(co_pop, int(math.ceil(algo_param_4*len(co_pop))))
            mu_pop = list(map(self.toolbox.clone, mu_pop))
            for i in mu_pop:
                if i in co_pop:
                    co_pop.remove(i)


            for child1, child2 in zip(co_pop[::2], co_pop[1::2]):
                self.toolbox.mate(child1, child2)
                for i in range(self.chromosome_length):
                    if self.chromosome_type[i] == 1:
                        child1[i] = int(child1[i])
                        child2[i] = int(child2[i])
                del child1.fitness.values
                del child2.fitness.values

            for mutant in mu_pop:
                self.custom_mutate(mutant)
                del mutant.fitness.values

                # Evaluate the crossover and mutated population
            total_pop = offspring + co_pop + mu_pop
            invalid_ind = [ind for ind in total_pop if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            # Storing the best individuals after each generation
            best_individual = tools.selBest(total_pop, 1)[0]
            best_indi_per_gen.append(list(best_individual))
            best_indi_fitness_values.append(best_individual.fitness.values[0])


            tot_time = (time.time() - st_time)/(60*60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
            b3 = pd.Series(timer, name = 'Time')
            best_ind_df = pd.concat([b1,b2,b3], axis=1)
            best_ind_df.to_csv('best_ind.csv',index=False)


        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])

        b1 = pd.Series(best_indi_per_gen, name = 'Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name = 'Fitness_values')
        best_ind_df = pd.concat([b1,b2,b3], axis=1)
        best_ind_df.to_csv('best_ind.csv',index=False)

    #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n" , best_ind_df
        print "\n \n" , self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (self.n_generations, best_ind)
        return best_ind_df, best_ind

