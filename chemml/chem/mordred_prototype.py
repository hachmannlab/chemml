import pandas as pd
from mordred import Calculator, descriptors
from rdkit.Chem import MolFromSmiles


df = pd.read_csv('pka_rddesc_cleaned.csv')
df


# Creating Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)


smis = df['SMILES'].tolist()


pand = calc.pandas([MolFromSmiles(s) for s in smis])


import numpy as np

pand = pand.select_dtypes([np.number]).replace([np.inf, -np.inf], np.nan)


invariant_features = []
threshold = 2
for each in pand.columns:
    if pand[each].nunique() < threshold:
        invariant_features.append(each)

pand.drop(columns=invariant_features, inplace=True)
pand


# Generate matrix of correlation values
corr_matrix = pand.corr().abs()
# Keep only upper triangle of values, since the correlation matrix is mirrored around the diagonal
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find columns that are highly correlated
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


pand = pand.drop(columns=to_drop)
pand


pand['SMILES'] = smis
pand


pand['pKa'] = df['pKa']
pand


from rdkit.Chem import MolToInchi

pand['inchi'] = pand['SMILES'].apply(lambda s: MolToInchi(MolFromSmiles(s)))
pand


pand['temp'] = df['temp']
pand


pand = pand[pand['pKa']<14][pand['pKa']>0]
pand


pand.to_csv('pka_morddata_cleaned.csv',index=False)


