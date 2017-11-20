from deap import base, creator, tools
import random
import numpy as np
import csv
import pandas

class GA_NN(object):
    def __init__(self, Evaluate, Weights, chromosome_length, chromosome_type, bit_limits, pop_size, crossover_prob, crossover_type, mutation_prob, mut_float_param_1, mut_float_param_2, mut_int_param_1, mut_int_param_2, selection_type, n_generations ):

	# Evaluate: The objective function that has to be optimized
	# Weights: A tuple containing fitness objective(s) for objective function(s). Ex: (1.0,) for maximizing and (-1.0,) for minimizing a single objective finction
	# chromosome_length: An integer which specifies the length of chromosome/individual
	# chromosome_type: A tuple of length chromosome_length describing the type of each bit of the chromosome. 0 for floating type and 1 for integer type
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

	creator.create("FitnessMin", base.Fitness, weights=self.Weights)
	creator.create("Individual", list, fitness=creator.FitnessMin)
	self.toolbox = base.Toolbox()
	
	def chromosome_generator():
	    A = []
    	    for i in range(self.chromosome_length):
		if self.chromosome_type[i] == 0:
		    A.append(random.uniform(self.bit_limits[i][0], self.bit_limits[i][1]))
		else:
		    A.append(random.randint(self.bit_limits[i][0], self.bit_limits[i][1]+1))
            return A

	self.toolbox.register("individual", tools.initIterate, creator.Individual, chromosome_generator)
	self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

	if self.crossover_type == "OnePoint":
	    self.toolbox.register("mate", tools.cxOnePoint)
	elif self.crossover_type == "TwoPoint":
	    self.toolbox.register("mate", tools.cxTwoPoint)
	elif self.crossover_type == "Uniform":
	    self.toolbox.register("mate", tools.cxUniform, indpb=self.crossover_prob)

	if self.selection_type == "Tournament":
	    self.toolbox.register("select", tools.selTournament, tournsize=20)
	elif self.selection_type == "Roulette":
	    self.toolbox.register("select", tools.selRoulette)

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
		    s += ((x-low)*10)**2
		elif x > up:
		    s += ((x-up)*10)**2
	    return s

	self.toolbox.register("evaluate", self.evaluate)
	self.toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, -100.0, distance))

    def custom_mutate(self, indi):
	for i in range(self.chromosome_length):
	    if self.chromosome_type[i] == 0:
	        if random.random() < self.mutation_prob:
		    indi[i] += random.gauss(self.mut_float_param_1, self.mut_float_param_2)
	    elif self.chromosome_type[i] == 1:
		if random.random() < self.mutation_prob:
		    indi[i] += random.randint(self.mut_int_param_1, self.mut_int_param_2)

    def main_func(self):
        pop = self.toolbox.population(n=self.pop_size)
	print pop
        # Evaluate the entire population
	fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
#        fits = [indi.fitness.values[0] for indi in pop]

	best_indi_per_gen = []
        for g in range(self.n_generations):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
		    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

	    for mutant in offspring:
		self.custom_mutate(mutant)
		del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
	    fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

	    # Storing the best individuals after each generation
	    best_indi_per_gen.append(list(tools.selBest(pop, 1)[0]))
            # The population is entirely replaced by the offspring
            pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]
	best_indi_per_gen.append(list(best_ind))
	best_ind_df = pandas.DataFrame(best_indi_per_gen)
	print "\n \n Best Individuals of each generation are:  \n \n" , best_ind_df
        print "\n \n" , self.bit_limits, " \n \n Best individual after %s evolutions is %s " % (self.n_generations, best_ind)
	""" 
	# This is optional if you want it to write to a csv file
	writer = csv.writer(open('best_inds.csv' , 'w') , delimiter = ',')
	### write header for the dataframe here <--
	for ind,row in best_ind_df.iterrows():
    	    writer.writerow(row)
	"""


def eval_sum(individual):
    # For a single objective function as this one, PUT a comma at the end to make this function return a tuple!!!
    return sum(individual),

Weights = (1.0, )
chromosome_length = 4
chromosome_type = (1,1,1,0)
bit_limits = [(0,9) , (0,20) , (1,16) , (0.00001,1) ]
pop_size = 100
crossover_prob = 0.45
mutation_prob = 0.15
crossover_type = "OnePoint"
mut_float_param_1 = 0
mut_float_param_2 = 1
mut_int_param_1 = 1
mut_int_param_2 = 3
selection_type = "Tournament"
n_generations = 500

GV = GA_NN( eval_sum, Weights, chromosome_length, chromosome_type, bit_limits, pop_size, crossover_prob, crossover_type, mutation_prob, mut_float_param_1, mut_float_param_2, mut_int_param_1, mut_int_param_2, selection_type, n_generations )

GV.main_func()

