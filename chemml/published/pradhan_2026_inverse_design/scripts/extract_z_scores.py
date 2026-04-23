import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator # <-- The correct modern import

start_time = time.time()

def z_score(k, K, m, M):
    """ k = number of occurences in m
        K = number of occurences in full dataset
        m = subset size
        M = full data set size """
    # Calculate hypergeometric variance
    sig_i = (m * K / M) * ((M - K) / M) * ((M - m) / (M - 1))
    
    # Safety check: Avoid division by zero if a bit has zero variance
    if sig_i <= 0:
        return 0
        
    z = (k - (m * K / M)) / np.sqrt(sig_i)
    return z

def get_fragment_smiles(mol, atom_idx, radius):
    """Helper function to extract the SMILES of a specific Morgan fragment"""
    # If the radius is 0, it's just a single atom
    if radius == 0:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[atom_idx])
    
    # If radius > 0, find the surrounding environment
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms_to_use = set()
    for b in env:
        bond = mol.GetBondWithIdx(b)
        atoms_to_use.add(bond.GetBeginAtomIdx())
        atoms_to_use.add(bond.GetEndAtomIdx())
        
    if not atoms_to_use:
        return None
    return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms_to_use))

# --- 1. Load and Prepare Data ---
print("Loading data...")
raw_df_1 = pd.read_csv("../data/100k_den.csv")
raw_df_2 = pd.read_csv("../data/paper9_MF.csv")
raw_df = pd.concat([raw_df_1, raw_df_2], axis=1)

print(f"raw_df shape: {raw_df.columns}")
df = raw_df.copy()

# Identify columns (Assumes Col 0 is SMILES, Col 1 is RI, Col 2+ are bits)
smiles_col = df.columns[0]
target_col = df.columns[1]
bit_cols = df.columns[2:]

# Sort by target property (descending) and reset index
df = df.sort_values(by=target_col, ascending=False).reset_index(drop=True)

M = df.shape[0]
m = int(0.1 * M) # Top 10% subset
print(f"M (Total dataset): {M}")
print(f"m (Top 10% subset): {m}")

# --- 2. Calculate k and K using Vectorization (Much Faster) ---
print("Starting K and k calculations...")
K_vals = df[bit_cols].sum().values       
k_vals = df.loc[:m-1, bit_cols].sum().values 

# --- 3. Calculate Z-scores ---
print("Starting z-score calculations...")
z_scores = [z_score(k, K, m, M) for k, K in zip(k_vals, K_vals)]

# Create a clean DataFrame of the results
z_df = pd.DataFrame({
    'Bit': bit_cols,
    'Z_Score': z_scores,
    'k': k_vals,
    'K': K_vals
})

# Save full results to a proper CSV file
# z_df.to_csv("../outputs/z_scores.csv", index=False)

# Get the top 30 bits
top_30_bits = z_df.sort_values(by='Z_Score', ascending=False).head(30)

# --- 4. Extract Fragment SMILES for Top 30 Bits (USING THE NEW API) ---
print("\nExtracting SMILES for the top 30 Morgan bits...")
top_30_results = []

# Initialize the new generator once outside the loop
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

for index, row in top_30_bits.iterrows():
    bit_name = row['Bit']
    z_val = row['Z_Score']
    
    # Parse the integer bit index from the column name (e.g., "bit_54" -> 54)
    try:
        bit_idx = int(str(bit_name).replace('bit_', ''))
    except ValueError:
        bit_idx = int(bit_name) 
        
    # Find one example molecule in the top 10% that contains this bit
    example_row = df.loc[:m-1][df.loc[:m-1, bit_name] == 1].head(1)
    
    if example_row.empty:
        fragment_smiles = "Not found"
    else:
        smi = example_row[smiles_col].values[0]
        mol = Chem.MolFromSmiles(smi)
        
        # --- THE FIX: Using AdditionalOutput to extract bitInfo ---
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        
        # Pass the 'ao' object into the generator call
        fp = mfpgen.GetFingerprint(mol, additionalOutput=ao)
        
        # Extract the dictionary
        bit_info = ao.GetBitInfoMap()
        # ----------------------------------------------------------
        
        if bit_idx in bit_info:
            # Take the first occurrence of this bit in the molecule
            atom_idx, radius = bit_info[bit_idx][0]
            fragment_smiles = get_fragment_smiles(mol, atom_idx, radius)
        else:
            fragment_smiles = "Hashing collision error"
            
    top_30_results.append((bit_name, z_val, fragment_smiles))

# --- 5. Print Output ---
print("\n" + "="*60)
print(f"{'Rank':<5} | {'Bit':<8} | {'Z-Score':<8} | {'Fragment SMILES'}")
print("="*60)
for rank, (bit, z, frag_smi) in enumerate(top_30_results, 1):
    print(f"{rank:<5} | {bit:<8} | {z:<8.2f} | {frag_smi}")
    
print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")