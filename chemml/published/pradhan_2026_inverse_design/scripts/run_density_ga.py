import zipfile
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from chemml.optimization import GeneticAlgorithm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

output_dir = "outputs/"
start_time = time.time()

with open("data/den_bb.txt", "r") as f:
    X = f.readlines()

linkers = [str(i.strip()) for i in X[:6]]
hetero = [str(i.strip()) for i in X[6:]]

print("Building blocks read!")

# --- 2. LOAD & PREPARE DATA ---
# Column 2 onwards are the Morgan FP bits
raw_df_1 = pd.read_csv("data/100k_den.csv")
with zipfile.ZipFile("data/paper9_MF.csv.zip", "r") as z:
    with z.open("paper9_MF.csv") as f:
        raw_df_2 = pd.read_csv(f)
raw_df = pd.concat([raw_df_1, raw_df_2], axis=1)

# Drop any potential duplicate indices/columns if necessary
X_train = raw_df.iloc[:, 2:]
y_train = raw_df.iloc[:, 1]

# --- 3. TRAIN THE SURROGATE MODEL ---
# Using the exact parameters from your manuscript's Methods/Results section
alpha = 2.74e-5
learning_rate_init = 9.04e-5
activation = 'tanh'
hidden_layer_sizes = (256, 256, 64)

print("Starting ML fit for Density surrogate...")
regr = MLPRegressor(
    alpha=alpha, 
    learning_rate_init=learning_rate_init, 
    activation=activation, 
    hidden_layer_sizes=hidden_layer_sizes
)
regr.fit(X_train, y_train)
print("Model training complete!")

# --- 4. RDKIT SETUP ---
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def single_obj(smiles):
    m = Chem.MolFromSmiles(smiles)    
    
    # Penalize chemically invalid structures (shattered graphs)
    if m is None:
        return np.array([-999.0])
        
    fp = mfpgen.GetFingerprint(m)
    fp_array = np.array(fp)
    
    # Predict density
    y_predict = regr.predict([fp_array])
    return y_predict
    
def ga_eval(indi):
    new_mol = ""
    for fragment in indi[:-1]:  # Exclude the dummy variable at the end
        new_mol += fragment
    
    ga_search = single_obj(new_mol)
    return ga_search

# --- 5. GENETIC ALGORITHM SPACE ---
space = (
    {'F1': {'choice': hetero}},
    {'F2': {'choice': linkers}},
    {'F3': {'choice': hetero}},    
    {'F4': {'choice': linkers}},
    {'F5': {'choice': hetero}},
    {'F6': {'choice': linkers}},
    {'F7': {'choice': hetero}},
    {'F8': {'choice': linkers}},
    {'F9': {'choice': hetero}},
    {'F10': {'choice': linkers}},
    {'F11': {'choice': hetero}},
    {'dummy': {'uniform': (0, 1), 'mutation': [0, 1]}}
)              
                    
# --- 6. RUN GA ---
print("Starting Genetic Algorithm search...")
gann = GeneticAlgorithm(
    evaluate=ga_eval, 
    space=space, 
    fitness=('max',), 
    pop_size=40, 
    crossover_size=100, 
    mutation_size=100, 
    algorithm=1
)

best_ind_df, best_individual = gann.search(n_generations=200, early_stopping=50)  
# best_ind_df, best_individual = gann.search(n_generations=2, early_stopping=50)                     
#                    
print("Genetic Algorithm - complete")

# --- 7. SAVE OUTPUTS ---
all_items = list(gann.fitness_dict.items())
all_items_df = pd.DataFrame(all_items, columns=['moeties', 'Density'])

all_items_df.to_csv(f'{output_dir}fitness_dict_den_output.csv')
best_ind_df.to_csv(f'{output_dir}ga_best_candidates_den_output.csv')

print(f"\nBest particle: {best_individual}")
print("----------------%s seconds ------------------" % (time.time() - start_time))