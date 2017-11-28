from deap import base, creator, tools
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import multiprocessing
import time
from sklearn.model_selection import cross_val_score

import math

start_time = time.time()


class GA_NN(object):
    def __init__(self, Evaluate, algorithm, Weights, chromosome_length, chromosome_type, bit_limits, pop_size,
                 crossover_prob, crossover_type, mutation_prob, mut_float_param_1, mut_float_param_2, mut_int_param_1,
                 mut_int_param_2, selection_type, n_generations):

        # Evaluate: The objective function that has to be optimized
        # Weights: A tuple containing fitness objective(s) for objective function(s). Ex: (1.0,) for maximizing and (-1.0,) for minimizing a single objective finction
        # chromosome_length: An integer which specifies the length of chromosome/individual
        # chromosome_type: A tuple of length chromosome_length describing the type of each bit of the chromosome. 0 for floating type and 1 for integer type. Keep all integer types first followed by the floating types.
        # bit_limits: A list of tuples containing the lower and upper limits for each bit
        # pop_size: An integer which denotes the size of the population (keep this > 50 always)
        # crossover_prob: A floating point number that denotes the Probability of crossover
        # crossover_type: A string denoting the type of crossover: Options are: OnePoint, TwoPoint or Uniform
        # mutation_prob: A floating point number that denotes the probability of mutation. Should be generally very low.
        # mut_float_param_1: Value of the mean of the Gaussian distribution for the Gaussian type mutation
        # mut_float_param_1: Value of the standard deviation of the Gaussian distribution for the Gaussian type mutation
        # mut_int_param_1: Value of the lower limit (inclusive) of the integer which is to be used for integer type mutation
        # mut_int_param_2: Value of the upper limit (inclusive) of the integer which is to be used for integer type mutation
        # selection_type: A string denoting the type of selection: Options are: Tournament or Roulette

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
        self.selection_type = selection_type
        self.n_generations = n_generations
        self.algo_choice = algorithm

	def fit(self):
        creator.create("FitnessMin", base.Fitness, weights=self.Weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()

        #	self.pool = multiprocessing.Pool(2)
        #	self.toolbox.register("map" , self.pool.map)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, chromosome_generator)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        """
        if self.crossover_type == "OnePoint":
            self.toolbox.register("mate", tools.cxOnePoint)
        elif self.crossover_type == "TwoPoint":
            self.toolbox.register("mate", tools.cxTwoPoint)
        elif self.crossover_type == "Uniform":
            self.toolbox.register("mate", tools.cxUniform, indpb=self.crossover_prob)
        """
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)

        """
        if self.selection_type == "Tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=30)
        elif self.selection_type == "Roulette":
            self.toolbox.register("select", tools.selRoulette)
        """
        self.toolbox.register("selectTournament", tools.selTournament, tournsize=30)
        self.toolbox.register("selectRoulette", tools.selRoulette)

        def feasibility(indi):
            for x, i in zip(indi, range(self.chromosome_length)):
                if self.bit_limits[i][0] <= x <= self.bit_limits[i][1]:
                    continue
                else:
                    return False
            return True

        def distance(indi):
            s = 0
            for x, i in zip(indi, range(self.chromosome_length)):
                low = self.bit_limits[i][0]
                up = self.bit_limits[i][1]
                if x < low:
                    s += ((x - low) * 10) ** 2
                elif x > up:
                    s += ((x - up) * 10) ** 2
            return s

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, -100.0 * self.Weights[0], distance))


	def chromosome_generator(self):
		A = []
		for i in range(self.chromosome_length):
			if self.chromosome_type[i] == 0:
				A.append(random.uniform(self.bit_limits[i][0], self.bit_limits[i][1]))
			else:
				A.append(random.randint(self.bit_limits[i][0], self.bit_limits[i][1]))
		return A


	def custom_mutate(self, indi):
			for i in range(self.chromosome_length):
				if self.chromosome_type[i] == 0:
					if random.random() < self.mutation_prob:
						indi[i] += random.gauss(self.mut_float_param_1, self.mut_float_param_2)
				elif self.chromosome_type[i] == 1:
					if random.random() < self.mutation_prob:
						indi[i] = random.randint(self.mut_int_param_1[i], self.mut_int_param_2[i])

    def main_func(self):
        if self.algo_choice == 1:
            self.algo1()
        elif self.algo_choice == 2:
            self.algo2()
        elif self.algo_choice == 3:
            self.algo3()
        elif self.algo_choice == 4:
            self.algo4()

    def algo1(self):
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8 * len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3 * len(pop))))
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

        # fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        MAE_test_list = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            offspring = tools.selBest(total_pop, self.pop_size)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8 * len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3 * len(pop))))
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

            act_list = ('identity', 'logistic', 'tanh')
            mlp = MLPRegressor(hidden_layer_sizes=(best_individual[0], best_individual[1], best_individual[2]), \
                               activation=act_list[int(best_individual[3]) - 1], alpha=best_individual[4],
                               learning_rate='invscaling', max_iter=100000)
            global train_indices, test_indices, xdata, y_final, clf
            mlp.fit(xdata[train_indices], y_final[train_indices])
            y_pred = mlp.predict(xdata[test_indices])
            y_pred = clf.inverse_transform(y_pred)
            y_test = clf.inverse_transform(y_final[test_indices])

            MAE_test = mean_absolute_error(y_test, y_pred)
            MAE_test_list.append(MAE_test)
            tot_time = (time.time() - st_time) / (60 * 60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
            b3 = pd.Series(MAE_test_list, name='MAE_test_set')
            b4 = pd.Series(timer, name='Time')
            best_ind_df = pd.concat([b1, b2, b3, b4], axis=1)
            best_ind_df.to_csv('best_a1.csv', index=False)

        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        act_list = ('identity', 'logistic', 'tanh')
        mlp = MLPRegressor(hidden_layer_sizes=(best_ind[0], best_ind[1], best_ind[2]), \
                           activation=act_list[int(best_ind[3]) - 1], alpha=best_ind[4], learning_rate='invscaling',
                           max_iter=100000)
        global train_indices, test_indices, xdata, y_final, clf
        mlp.fit(xdata[train_indices], y_final[train_indices])
        y_pred = mlp.predict(xdata[test_indices])
        y_pred = clf.inverse_transform(y_pred)
        y_test = clf.inverse_transform(y_final[test_indices])
        MAE_test = mean_absolute_error(y_test, y_pred)
        MAE_test_list.append(MAE_test)

        b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
        b3 = pd.Series(MAE_test_list, name='MAE_test_set')

        best_ind_df = pd.concat([b1, b2, b3], axis=1)
        best_ind_df.to_csv('best_a1.csv', index=False)

        #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n", best_ind_df
        print "\n \n", self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (
        self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algo2(self):
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8 * len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3 * len(pop))))
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

        # fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        MAE_test_list = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            off_pop = tools.selBest(pop, int(math.ceil(0.4 * len(pop))))
            off_co = tools.selBest(co_pop, int(math.ceil(0.3 * len(pop))))
            off_mu = tools.selBest(mu_pop, len(pop) - len(off_pop) - len(off_co))
            offspring = off_pop + off_co + off_mu
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8 * len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3 * len(pop))))
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

            act_list = ('identity', 'logistic', 'tanh')
            mlp = MLPRegressor(hidden_layer_sizes=(best_individual[0], best_individual[1], best_individual[2]), \
                               activation=act_list[int(best_individual[3]) - 1], alpha=best_individual[4],
                               learning_rate='invscaling', max_iter=100000)
            global train_indices, test_indices, xdata, y_final, clf
            mlp.fit(xdata[train_indices], y_final[train_indices])
            y_pred = mlp.predict(xdata[test_indices])
            y_pred = clf.inverse_transform(y_pred)
            y_test = clf.inverse_transform(y_final[test_indices])

            MAE_test = mean_absolute_error(y_test, y_pred)
            MAE_test_list.append(MAE_test)
            tot_time = (time.time() - st_time) / (60 * 60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
            b3 = pd.Series(MAE_test_list, name='MAE_test_set')
            b4 = pd.Series(timer, name='Time')
            best_ind_df = pd.concat([b1, b2, b3, b4], axis=1)
            best_ind_df.to_csv('best_a2.csv', index=False)

        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        act_list = ('identity', 'logistic', 'tanh')
        mlp = MLPRegressor(hidden_layer_sizes=(best_ind[0], best_ind[1], best_ind[2]), \
                           activation=act_list[int(best_ind[3]) - 1], alpha=best_ind[4], learning_rate='invscaling',
                           max_iter=100000)
        global train_indices, test_indices, xdata, y_final, clf
        mlp.fit(xdata[train_indices], y_final[train_indices])
        y_pred = mlp.predict(xdata[test_indices])
        y_pred = clf.inverse_transform(y_pred)
        y_test = clf.inverse_transform(y_final[test_indices])
        MAE_test = mean_absolute_error(y_test, y_pred)
        MAE_test_list.append(MAE_test)

        b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
        b3 = pd.Series(MAE_test_list, name='MAE_test_set')

        best_ind_df = pd.concat([b1, b2, b3], axis=1)
        best_ind_df.to_csv('best_a2.csv', index=False)

        #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n", best_ind_df
        print "\n \n", self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (
        self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algo3(self):
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8 * len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.3 * len(pop))))
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

        # fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        MAE_test_list = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            #	    offspring = tools.selBest(total_pop, self.pop_size)
            temp_pop = total_pop
            offspring = []
            for i in range(len(pop)):
                off = self.toolbox.selectRoulette(temp_pop, 1)
                offspring.append(off)
                temp_pop.remove(off)
            # Clone the selected individuals
            #            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8 * len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.3 * len(pop))))
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

            act_list = ('identity', 'logistic', 'tanh')
            mlp = MLPRegressor(hidden_layer_sizes=(best_individual[0], best_individual[1], best_individual[2]), \
                               activation=act_list[int(best_individual[3]) - 1], alpha=best_individual[4],
                               learning_rate='invscaling', max_iter=100000)
            global train_indices, test_indices, xdata, y_final, clf
            mlp.fit(xdata[train_indices], y_final[train_indices])
            y_pred = mlp.predict(xdata[test_indices])
            y_pred = clf.inverse_transform(y_pred)
            y_test = clf.inverse_transform(y_final[test_indices])

            MAE_test = mean_absolute_error(y_test, y_pred)
            MAE_test_list.append(MAE_test)
            tot_time = (time.time() - st_time) / (60 * 60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
            b3 = pd.Series(MAE_test_list, name='MAE_test_set')
            b4 = pd.Series(timer, name='Time')
            best_ind_df = pd.concat([b1, b2, b3], axis=1)
            best_ind_df.to_csv('best_a3.csv', index=False)

        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        act_list = ('identity', 'logistic', 'tanh')
        mlp = MLPRegressor(hidden_layer_sizes=(best_ind[0], best_ind[1], best_ind[2]), \
                           activation=act_list[int(best_ind[3]) - 1], alpha=best_ind[4], learning_rate='invscaling',
                           max_iter=100000)
        global train_indices, test_indices, xdata, y_final, clf
        mlp.fit(xdata[train_indices], y_final[train_indices])
        y_pred = mlp.predict(xdata[test_indices])
        y_pred = clf.inverse_transform(y_pred)
        y_test = clf.inverse_transform(y_final[test_indices])
        MAE_test = mean_absolute_error(y_test, y_pred)
        MAE_test_list.append(MAE_test)

        b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
        b3 = pd.Series(MAE_test_list, name='MAE_test_set')

        best_ind_df = pd.concat([b1, b2, b3], axis=1)
        best_ind_df.to_csv('best_a3.csv', index=False)

        #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n", best_ind_df
        print "\n \n", self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (
        self.n_generations, best_ind)
        return best_ind_df, best_ind

    def algo4(self):
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        co_pop = self.toolbox.selectRoulette(pop, int(math.ceil(0.8 * len(pop))))
        co_pop = list(map(self.toolbox.clone, co_pop))
        mu_pop = self.toolbox.selectRoulette(co_pop, int(math.ceil(0.4 * len(co_pop))))
        mu_pop = list(map(self.toolbox.clone, mu_pop))
        print co_pop, "ooooooooooooooo", mu_pop
        for i in mu_pop:
            print i, "oooooooooooo", mu_pop, "ooooooooooooo", co_pop
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

        # fits = [indi.fitness.values[0] for indi in pop]

        best_indi_per_gen = []
        best_indi_fitness_values = []
        MAE_test_list = []
        timer = []
        for g in range(self.n_generations):
            st_time = time.time()
            # Select the next generation individuals
            offspring = tools.selBest(total_pop, self.pop_size)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            co_pop = self.toolbox.selectRoulette(offspring, int(math.ceil(0.8 * len(pop))))
            co_pop = list(map(self.toolbox.clone, co_pop))
            mu_pop = self.toolbox.selectRoulette(co_pop, int(math.ceil(0.4 * len(co_pop))))
            mu_pop = list(map(self.toolbox.clone, mu_pop))
            for i in mu_pop:
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

            act_list = ('identity', 'logistic', 'tanh')
            mlp = MLPRegressor(hidden_layer_sizes=(best_individual[0], best_individual[1], best_individual[2]), \
                               activation=act_list[int(best_individual[3]) - 1], alpha=best_individual[4],
                               learning_rate='invscaling', max_iter=100000)
            global train_indices, test_indices, xdata, y_final, clf
            mlp.fit(xdata[train_indices], y_final[train_indices])
            y_pred = mlp.predict(xdata[test_indices])
            y_pred = clf.inverse_transform(y_pred)
            y_test = clf.inverse_transform(y_final[test_indices])

            MAE_test = mean_absolute_error(y_test, y_pred)
            MAE_test_list.append(MAE_test)
            tot_time = (time.time() - st_time) / (60 * 60)
            timer.append(tot_time)

            b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
            b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
            b3 = pd.Series(MAE_test_list, name='MAE_test_set')
            b4 = pd.Series(timer, name='Time')
            best_ind_df = pd.concat([b1, b2, b3, b4], axis=1)
            best_ind_df.to_csv('best_a4.csv', index=False)

        best_ind = tools.selBest(total_pop, 1)[0]
        best_indi_per_gen.append(list(best_ind))
        best_indi_fitness_values.append(best_ind.fitness.values[0])
        act_list = ('identity', 'logistic', 'tanh')
        mlp = MLPRegressor(hidden_layer_sizes=(best_ind[0], best_ind[1], best_ind[2]), \
                           activation=act_list[int(best_ind[3]) - 1], alpha=best_ind[4], learning_rate='invscaling',
                           max_iter=100000)
        global train_indices, test_indices, xdata, y_final, clf
        mlp.fit(xdata[train_indices], y_final[train_indices])
        y_pred = mlp.predict(xdata[test_indices])
        y_pred = clf.inverse_transform(y_pred)
        y_test = clf.inverse_transform(y_final[test_indices])
        MAE_test = mean_absolute_error(y_test, y_pred)
        MAE_test_list.append(MAE_test)

        b1 = pd.Series(best_indi_per_gen, name='Best_individual_per_gen')
        b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
        b3 = pd.Series(MAE_test_list, name='MAE_test_set')

        best_ind_df = pd.concat([b1, b2, b3], axis=1)
        best_ind_df.to_csv('best_a4.csv', index=False)

        #	best_ind_df = pd.DataFrame(best_indi_per_gen)
        print "\n \n Best Individuals of each generation are:  \n \n", best_ind_df
        print "\n \n", self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (
        self.n_generations, best_ind)
        return best_ind_df, best_ind


xdata = pd.read_csv(
    '/projects/academic/hachmann/mojtaba/cheml/CheML/benchmarks/RI_project/liq_org/descriptors/MACCS_bit.csv',
    header=None)
ydata = pd.read_csv('/projects/academic/hachmann/mojtaba/cheml/CheML/benchmarks/RI_project/liq_org/pol_den_RI.csv')
#            ['RI_LL']
#              ['Pol_DFT']
#                  ['Den_MD']
xdata = xdata.loc[:, (xdata != xdata.ix[0]).any()]
xdata = xdata.values

from sklearn.preprocessing import StandardScaler

y = ydata['RI_LL']
clf = StandardScaler()
y_final = clf.fit_transform(y)
indices = range(len(y_final))

Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(xdata, y_final, indices, test_size=0.1)
with open('indices.py', 'w') as f:
    f.write('train_indices = %s' % train_indices)
    f.write('\n')
    f.write('test_indices = %s' % test_indices)

all_indi = []
all_MAE = []


	def eval_sum(individual, X=Xtrain[:50], y=ytrain[:50], clf=clf):
    # For a single objective function as this one, PUT a comma at the end to make this function return a tuple!!!
    act_list = ('identity', 'logistic', 'tanh')
    mlp = MLPRegressor(hidden_layer_sizes=(individual[0], individual[1], individual[2]), \
                       activation=act_list[int(individual[3]) - 1], alpha=individual[4], learning_rate='invscaling',
                       max_iter=100000)
    """

    kf = KFold(n_splits = 5)
    cv_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mlp.fit(X_train,y_train)
        y_pred = mlp.predict(X_test)
        y_pred = clf.inverse_transform(y_pred)
    y_test = clf.inverse_transform(y_test)
    score = mean_absolute_error(y_test,y_pred)
        cv_scores.append(score)
    score = np.mean(cv_scores)
    score = 1000*score + 0.01*sum(individual[0:3])
    return score,
    """
    MAE_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(estimator=mlp, X=X, y=y, scoring=MAE_scorer, cv=5, n_jobs=6)
    score = np.mean(scores)  ## NOT INVERSE TRANSFORMED
    #    score = 1000*score + 0.01*sum(individual[0:3])
    #    return score, sum(individual[0:3])
    global all_indi
    all_indi.append(individual)
    global all_MAE
    all_MAE.append(-1 * score)
    a1 = pd.Series(all_indi, name='Individual')
    a2 = pd.Series(all_MAE, name='MAE_train_set_individuals')
    all_df = pd.concat([a1, a2], axis=1)
    all_df.to_csv('all_indi_MAE.csv', index=False)
    return -1 * score,


Weights = (-1.0,)
chromosome_length = 5
chromosome_type = (1, 1, 1, 1, 0)
bit_limits = [(10, 250), (10, 250), (10, 250), (1, 3), (0.0001, 1)]
pop_size = 10
crossover_prob = 0.4
mutation_prob = 0.4
crossover_type = "TwoPoint"
mut_float_param_1 = 0
mut_float_param_2 = 1
mut_int_param_1 = (10, 10, 10, 1)
mut_int_param_2 = (250, 250, 250, 3)
selection_type = "Tournament"
n_generations = 2
algor = 4

GV = GA_NN(eval_sum, algor, Weights, chromosome_length, chromosome_type, bit_limits, pop_size, crossover_prob,
           crossover_type, mutation_prob, mut_float_param_1, mut_float_param_2, mut_int_param_1, mut_int_param_2,
           selection_type, n_generations)

df, individual = GV.main_func()

print "\n \n \n MAE of test set: ", score
total_time = (time.time() - start_time) / (60 * 60)
print " \n \n \n Total execution time in hours: ", total_time
