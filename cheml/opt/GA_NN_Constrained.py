import math
import random
import numpy as np
import copy
import pickle as pkl
import pandas as pd

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
    individuals: integer or list of rings
        if integer, shows the number of rings in each chromosome.
        if list, shows a list of rings like 'R1','R2', ... that must be in the chromosome.
    n_pairs: integer
        number of pairs in the chromosome (before repairing)
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
    element_limit: integer, optional (default=1)
        maximum number of one elements of each chromosome
    n_iterations: integer, optional (default = 1000)
        Total number of iteration
    """
    def __init__(self, nnet, target, individuals, n_pairs,  nVar, normalizer = [0,0], crossover = 'Uniform',
                 selection = 'RWS', selection_pressure = 3, init_pop=100, crossover_ratio = 0.8,
                 mutation_ratio = 0.3, mutation_rate = 0.02, tournament_size = 3, element_limit = 1,
                 n_iterations = 1000):
        self.nnet = nnet
        self.target = target
        self.individuals = individuals
        self.n_pairs = n_pairs
        self.normalizer = normalizer
        self.nVar = nVar
        self.crossover = crossover
        self.selection = selection
        self.selection_pressure = selection_pressure
        self.init_pop = init_pop
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.element_limit = element_limit
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
            choices = range(self.element_limit+1)
            choices.remove(y[j[i]])
            y[j[i]] = random.choice(choices)
        return tuple(y)

    def CostFunction(self,x):
        prediction = nn.output(np.array(x),self.nnet)
        prediction = (self.normalizer[0] * prediction) + self.normalizer[1] #(0:std, 1:mean)
        error = np.mean(np.abs(prediction - np.array(self.target)))
        return error

    def chromosome2vector(self,chromosome):
        x = [0] * len(self.pairs)
        for item in chromosome:
            x[item] += 1
        return x

    def constraints(self,c):
        bonds_matrix = pd.DataFrame()
        for pair_ind in c:
            
        return c

    def build_chromosome(self,selected_rings,n_chromosome):
        selected_rings.sort()
        for i in range(n_chromosome):
            if selected_rings == self.rings:
                self.selected_pairs = copy.deepcopy(self.pairs)
            else:
                temp = []
                for ring in selected_rings:
                    temp+=self.info[ring]
                self.selected_pairs = [key for key in temp if temp.count(key)>1]
                for ring in selected_rings:
                    if self.L[ring][ring]==1:
                        self.selected_pairs.append(self.pairs.index('%s_%s'%(ring,ring)))
                    if self.F[ring][ring]==1:
                        self.selected_pairs.append(self.pairs.index('%s*%s'%(ring,ring)))
            self.selected_pairs = list(set(self.selected_pairs))
            chromosome = random.sample(self.selected_pairs,self.n_pairs)
            chromosome = self.constraints(chromosome)
            self.pop.append(chromosome)
            x = self.chromosome2vector(chromosome)
            self.cost.append(self.CostFunction(x))

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
                q = random.randint(0,self.init_pop-1)
                r = random.randint(0,self.init_pop-1)
            else:
                msg = "'%s' is not a valid selection method"%self.selection
                raise NameError(msg)
            # Parents
            p1 = self.pop[q]
            p2 = self.pop[r]

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
                q = random.randint(0,self.init_pop-1)
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
        with open(self.nnet, 'rb') as network:
            self.nnet = pkl.load(network)
        self.rings = ['R1', 'R10', 'R11', 'R12', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
        self.pairs = ['R1*R1', 'R1_R1', 'R1*R10', 'R1_R10', 'R1_R11', 'R1_R12', 'R1*R2', 'R1_R2', 'R1*R3', 'R1_R3',
                   'R1*R4', 'R1_R4', 'R1*R5', 'R1_R5', 'R1*R6', 'R1_R6', 'R1*R7', 'R1_R7', 'R1*R9', 'R10*R10',
                   'R10_R10', 'R10_R11', 'R10_R12', 'R10*R2', 'R10_R2', 'R10*R3', 'R10_R3', 'R10*R4', 'R10_R4',
                   'R10*R5', 'R10_R5', 'R10*R6', 'R10_R6', 'R10*R7', 'R10_R7', 'R10*R9', 'R11_R12', 'R11_R2', 'R11_R3',
                   'R11_R4', 'R11_R5', 'R11_R6', 'R11_R7', 'R12_R12', 'R12_R2', 'R12_R3', 'R12_R4', 'R12_R5', 'R12_R6',
                   'R12_R7', 'R2*R2', 'R2_R2', 'R2*R3', 'R2_R3', 'R2*R4', 'R2_R4', 'R2*R5', 'R2_R5', 'R2*R6', 'R2_R6',
                   'R2*R7', 'R2_R7', 'R2*R8', 'R3*R3', 'R3_R3', 'R3*R4', 'R3_R4', 'R3*R5', 'R3_R5', 'R3*R6', 'R3_R6',
                   'R3*R7', 'R3_R7', 'R4*R4', 'R4_R4', 'R4*R5', 'R4_R5', 'R4*R6', 'R4_R6', 'R4*R7', 'R4_R7', 'R5*R5',
                   'R5_R5', 'R5*R6', 'R5_R6', 'R5*R7', 'R5_R7', 'R6*R6', 'R6_R6', 'R6*R7', 'R6_R7', 'R7*R7', 'R7_R7']
        self.L = pd.DataFrame(index=self.rings,columns=self.rings) # linking matrix
        self.F = pd.DataFrame(index=self.rings,columns=self.rings) # fusion matrix
        self.pair_elements = pd.DataFrame(index=range(len(self.pairs)),columns=['B','Rs'])
        for i, pair in enumerate(self.pairs):
            if '_' in pair:
                Rs = pair.split('_')
                self.L[Rs[0]][Rs[1]] = 1
                self.L[Rs[1]][Rs[0]] = 1
                self.pair_elements['B'][i] = 'L'
                self.pair_elements['Rs'][i] = Rs
            elif '*' in pair:
                Rs = pair.split('*')
                self.F[Rs[0]][Rs[1]] = 1
                self.F[Rs[1]][Rs[0]] = 1
                self.pair_elements['B'][i] = 'F'
                self.pair_elements['Rs'][i] = Rs
        # L.fillna(-1, inplace=True)
        # F.fillna(-1, inplace=True)

        self.info = {r:[] for r in self.rings}
        self.info['R1'] = range(19)
        for ring in self.rings[1:]:
            for i, pair in enumerate(self.pairs):
                if ring in pair:
                    self.info[ring]+=[i]
        if isinstance(self.individulas, list):
            if set(self.individuals)<=self.rings:
                self.build_chromosome(self.individuals, self.init_pop)
            else:
                msg = 'not a valid list of individulas based on available rings:%s'%str(self.rings)
                raise ValueError(msg)
        elif isinstance(self.individuals,int):
            if self.individuals<=len(self.rings):
                for i in range(self.init_pop):
                    choices = random.sample(self.rings,self.individuals)
                    self.build_chromosome(choices,1)
            else:
                msg = 'maximum number of individuals is %i'%len(self.rings)
                raise ValueError(msg)
        else:
            msg = 'individulas must be a list of rings or number of individuals'
            raise ValueError(msg)
        for i in xrange (self.init_pop):
            # Initialize Position
            a=[]
            for i in range (0,self.nVar):
                q = random.randint(0,self.element_limit)
                a.append(q)
            a = tuple(a)
            self.pop.append(a)

            # Evaluation
            self.cost.append(self.CostFunction(a))

        ## Sort Population
        self.cost, self.pop = (list(t)for t in zip(*sorted(zip(self.cost, self.pop))))

        ## Store cost
        worstcost=[]
        worstcost.append(self.cost[-1])
        WorstCost=np.nanmax(worstcost)

        ## Main Loop
        for it in xrange (0,self.n_iterations+1):
            # Calculate Selection Probabilities
            P = np.exp(-self.selection_pressure*np.array(self.cost)/WorstCost)
            P = P / float(sum(P))
            popc, costc = self.Crossover(P)
            popm, costm = self.Mutation(P)
            # Create Merge Population & Cost
            self.pop = self.pop + popc + popm
            self.cost = self.cost + costc + costm

            self.pop = tuple(self.pop)
            self.cost = tuple(self.cost)

            # Sort Population
            self.cost, self.pop = (list(t)for t in zip(*sorted(zip(self.cost, self.pop))))

            # Update WorstCost
            worstcost.append(self.cost[-1])
            WorstCost = np.nanmax(worstcost)

            # Truncation
            self.pop = self.pop[0:self.init_pop]
            self.cost = self.cost[0:self.init_pop]

            # Store Best Solution Ever Found
            BestSol = self.pop[0]

            # Store Best Cost Ever Found
            BestCost = self.cost[0]


            # Save Iteration Information
            if it%100 == 0:
                with open('GA_Iteration_Records.txt','a') as it_file:
                    it_file.write('Iteration: %i, Best cost $ %s\n'%(it, str(BestCost)))
                    it_file.write('Iteration: %i, Best chromosome: %s\n'%(it, str(BestSol)))
                    it_file.write('          GA+NN by MHL\n')
