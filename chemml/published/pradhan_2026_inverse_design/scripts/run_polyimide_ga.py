import datetime
import time
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from chemml.optimization import GeneticAlgorithm
from sklearn.neural_network import MLPRegressor
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

output_dir = "outputs/"

start_time=time.time()
# read .txt file for building blocks
f = open("data/ri_bb.txt", "r")
X=f.readlines()
f.close()

pi_df=pd.read_csv(r"data/paper13_smiles_ri_mf.csv")

alpha=0.001
activation='tanh'
hidden_layer_sizes=(100,200,40)
print("starting ML fit")
regr = MLPRegressor(alpha=alpha, activation=activation,hidden_layer_sizes=hidden_layer_sizes)
regr.fit(pi_df.iloc[:,2:],pi_df.iloc[:,1])

print("Files read!")

linkers=[]
hetero=[]
for i in X[:6]:
    linkers.append(str(i.strip()))
for i in X[6:26]:
    hetero.append(str(i.strip()))

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def single_obj(smiles):
    m = Chem.MolFromSmiles(smiles)           
    
    # Optional safety check: If RDKit can't parse it, penalize it
    if m is None:
        return np.array([-999.0])
        
    fp = mfpgen.GetFingerprint(m)
    
    fp_array=np.array(fp)
    y_predict=regr.predict([fp_array])
    return y_predict
    
def ga_eval(indi):
    new_mol="N1C(=O)c2c(C1=O)ccc(c2)"    #start phthalamide
    for mol in indi[:5]:
        new_mol+=mol
    new_mol=new_mol+"c1ccc2c(c1)C(=O)N(C2=O)"   #middle phthalamide
    for mol in indi[5:-1]:
        new_mol+=mol
    
    ga_search = single_obj(new_mol)
    return ga_search

space = ({'R1A': {'choice': linkers}},
        {'R1B': {'choice': hetero}},
        {'R1C': {'choice': linkers}},    
        {'R1D': {'choice': hetero}},
        {'R1E': {'choice': linkers}},
        {'R2A': {'choice': hetero}},
        {'R2B': {'choice': linkers}},
        {'R2C': {'choice': hetero}},
        {'R2D': {'choice': linkers}},
        {'R2E': {'choice': hetero}},
        {'dummy':{'uniform':(0,1),
         'mutation': [0, 1]}}
        )              
                    
gann = GeneticAlgorithm(evaluate=ga_eval, space=space, fitness=('max',), pop_size = 40, crossover_size=100, mutation_size=100, algorithm=1)
# best_ind_df, best_individual = gann.search(n_generations=2, early_stopping=500)  
best_ind_df, best_individual = gann.search(n_generations=200, early_stopping=500)                     

print("GeneticAlgorithm - complete")

all_items = list(gann.fitness_dict.items())
all_items_df = pd.DataFrame(all_items, columns=['moeties', 'RI'])
all_items_df.to_csv(f'{output_dir}fitness_dict_pi_output.csv')
best_ind_df.to_csv(f'{output_dir}pi_output.csv')
print("\n\n\n\n\n\n\n\n\n\n\n\ngenetic algorithm: \n", best_ind_df, "\n\nbest particle: ", best_individual)
print("\n")
print("----------------%s seconds ------------------" % (time.time()-start_time))