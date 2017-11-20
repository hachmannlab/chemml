import math
import random
import numpy as np
import copy
import pickle as pkl
from ..nn.nn_psgd import
class GA_nn(object):
    """
    Genetic Algorithm mixed with a trained neural network as its objective function

    Parameters
    ----------
    nnet: string
        The path to the pickle file that contains the weights and activation functions of a
        trained neural network.
    target: list
        list of desired values for outputs
    nVar: integer
        Number of elements in each chromosome
    normalizer: list, optional (default = [0,0])
        A list of standard deviation (first element) and mean (second element) of target values
        in the training set.
    crossover: string, optinal (default = 'Uniform')
        A choice of three different crossover methods: 'SinglePoint', 'DoublePoint', 'Uniform'
    selection: string, optional (default = 'RWS')
        A choice of three different methods for the selection of parent chromosomes:
            'RWS' : Roulette Wheel Selection
            'TS' : Tournament Selection
            'RS' : Random Selection
    selection_pressure: integer, optional (default = 3)
        selection pressure determines the effect of cost of a chromosome in the chance of selection
    init_pop: integer, optional (default = 100)
        The initialized population of chromosomes
    crossover_ratio: float in the range(0,1), optional (default = 0.8)
        The proportion of total number of chromosomes to be selected for crossover
    mutation_ratio: float in the range(0,1), optional (default = 0.3)
        The proportion of total number of chromosomes to be selected for mutation
    mutaton_rate: float, optional (default = 0.02)
        The proportion of total number of elements in a chromosome for mutation.
    tournament_size: integer, optional (default = 3)
        size of subgroups for the tournament selection
    n_iterations: integer, optional (default = 1000)
        Total number of iteration
    """
    def __init__(self, nnet, target, nVar, normalizer = [0,0], crossover = 'Uniform', selection = 'RWS',
                 selection_pressure = 3, init_pop=100, crossover_ratio = 0.8, mutation_ratio = 0.3,
                 mutation_rate = 0.02, tournament_size = 3,n_iterations = 1000):
        self.nnet = nnet
        self.target = target
        self.normalizer = normalizer
        self.nVar = nVar3366
        self.crossover = crossover
        self.selection = selection
        self.selection_pressure = selection_pressure
        self.init_pop = init_pop
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.n_iterations = n_iterations

    def SinglePointCrossover(self, x1, x2):
        nVar=len(x1)
        c = random.randint(1,nVar-1)
        y1=x1[0:c]				# the start point and length of range(end-start) is important
        y1=y1+x2[c:nVar]      # the start point and length of range(end-start) is important
        y2=x2[0:c]
        y2=y2+x1[c:nVar]
        return tuple(y1), tuple(y2)

    def DoublePointCrossover(self, x1, x2):
        nVar = len(x1)
        cc = random.sample(range(1,nVar), 2)   # the start point and length of range(end-start) is important => here range is 1 to nVar-2
        c1 = min(cc)
        c2 = max(cc)
        y1 = x1[0:c1]+x2[c1:c2]+x1[c2:nVar]				# the start point and length of range(end-start) is important
        y2 = x2[0:c1]+x1[c1:c2]+x2[c2:nVar]      # the start point and length of range(end-start) is important
        return tuple(y1), tuple(y2)

    def UniformCrossover(self, x1, x2):
        nVar = len(x1)
        a = []
        for i in range (0,nVar):
            q = random.randint(0,1)
            a.append(q)
        alpha = np.array(a)
        y1 = alpha*x1+(1-alpha)*x2
        y2 = alpha*x2+(1-alpha)*x1
        return tuple(y1), tuple(y2)

    def RouletteWheelSelection(self, X):
        r = random.uniform(0,1)
        c=np.cumsum(X)
        c=np.array(c).tolist()
        f=0
        for i in range (0,len(c)-1):
            if r<=c[i]:
                f=i
                break
        return(f)

    def TournamentSelection(self, C, m):
        nVar=len(C)
        c=[]
        S = random.sample(range(0,nVar), m)
        for i in range (0,m):
            c.append(C[S[i]])
        c, S = (list(t)for t in zip(*sorted(zip(c, S))))
        return(S[0])

    def Mutate(self, x):
        nVar = len(x)
        nmu = math.ceil(self.mutation_rate*nVar)
        j = random.sample(range(nVar), int(nmu))
        y = copy.deepcopy(x)
        y = list(y)
        for i in xrange(int(nmu)):
            y[j[i]] = 1-y[j[i]]
        return tuple(y)

    def CostFunction(self,x):
        with open(self.nnet, 'rb') as network:
            self.nnet = pkl.load(network)
        prediction = nn.output(np.array(x),nnet)
        prediction = (self.normalizer[0] * prediction) + self.normalizer[1] #(0:std, 1:mean)
        error = np.mean(np.abs(prediction - np.array(self.target)))
        return error

    def Crossover(self,P):
        nc = 2 * math.ceil(self.crossover_ratio * self.init_pop / 2)
        popc = []
        costc = []
        for k in xrange(0, int(nc / 2)):
            if self.selection == 'RWS':
                q = self.RouletteWheelSelection(P)  # SP=0 is like random
                r = self.RouletteWheelSelection(P)  # SP=0 is like random
            elif self.selection == 'TS':
                q = self.TournamentSelection(self.cost, self.tournament_size)
                r = self.TournamentSelection(self.cost, self.tournament_size)
            elif self.selection == 'RS':
                q = random.randint(0,nPop-1)
                r = random.randint(0,nPop-1)
            else:
                msg = "'%s' is not a valid selection method"%self.selection
                raise NameError(msg)
            # Parents
            p1 = pop[q]
            p2 = pop[r]

            # Apply Crossover & Make popc
            if self.crossover == 'SinglePoint':
                a = self.SinglePointCrossover(p1,p2)       # SinglePointCrossover SPC
            elif self.crossover == 'DoublePoint':
                a = self.DoublePointCrossover(p1,p2)       # DoublePointCrossover DPC
            elif self.crossover == 'Uniform':
                a = self.UniformCrossover(np.array(p1), np.array(p2))          # UniformCrossover UC

            # Evaluate Offsprings & Make costc
            a1 = self.repair(a[0])
            popc.append(a1)
            costc.append(self.CostFunction(a1))
            a2 = self.repair(a[1])
            popc.append(a2)
            costc.append(self.CostFunction(a2))
        return popc, costc

    def Mutation(self,P):
        nm = math.ceil(self.mutation_ratio * self.init_pop)
        popm = []
        costm = []
        for k in range(0, int(nm)):
            if self.selection == 'RWS':
                q = self.RouletteWheelSelection(P)
            elif self.selection == 'TS':
                q = self.TournamentSelection(self.cost, self.tournament_size)
            elif self.selection == 'RS':
                q = random.randint(0,nPop-1)
            else:
                msg = "'%s' is not a valid selection method"%self.selection
                raise NameError(msg)
            # Mutant
            p = self.pop[q]

            # Apply Mutation & Make popm
            a = self.Mutate(p)
            a = self.repair(a)
            popm.append(a)

            # Evaluate Mutant and Make costm
            costm.append(self.CostFunction(np.array(a)))
        return popm, costm

    def fit(self):
        self.pop=[]
        self.cost=[]
        for i in xrange (0,self.init_pop):
            # Initialize Position
            a=[]
            for i in range (0,nvar):
                q = random.randint(0,1)
                a.append(q)
            a = tuple(a)
            self.pop.append(a)

            # Evaluation
            self.cost.append(self.CostFunction(a))

        ## Sort Population
        self.cost, self.pop = (list(t)for t in zip(*sorted(zip(self.cost, self.pop))))

        ## Store cost
        worstcost=[]
        worstcost.append(cost[-1])
        WorstCost=np.nanmax(worstcost)

        ## Main Loop
        for it in xrange (0,self.n_iterations):
            # Calculate Selection Probabilities
            P = np.exp(-self.selection_pressure*np.array(cost)/WorstCost)
            P = P / float(sum(P))
            popc, costc = self.Crossover(P)
            popm, costm = self.Mutation(P)
            # Create Merge Population & Cost
            pop = pop + popc + popm
            cost = cost + costc + costm

            pop = tuple(pop)
            cost = tuple(cost)

            # Sort Population
            cost, pop = (list(t)for t in zip(*sorted(zip(cost, pop))))

            # Update WorstCost
            worstcost.append(cost[-1])
            WorstCost = np.nanmax(worstcost)

            # Truncation
            pop = pop[0:nPop]
            cost = cost[0:nPop]

            # Store Best Solution Ever Found
            BestSol = pop[0]

            # Store Best Cost Ever Found
            BestCost = cost[0]


            # Save Iteration Information
            with open('GA_Iteration_Records.txt','a') as it_file:
                it_file.write('Iteration: %i, Best cost $ %s\n'%(it, str(BestCost)))
                it_file.write('Iteration: %i, Best chromosome: %s\n'%(it, str(BestSol)))
                it_file.write('          GA+NN by MHL\n')




