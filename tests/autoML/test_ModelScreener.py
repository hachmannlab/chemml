import pytest
import pandas as pd
import numpy as np
from chemml.autoML import ModelScreener
from chemml.chem import Molecule
from chemml.datasets import load_organic_density

x1=[]
y1=[]
for i in range(0, 10):
    x1.append(i)
    y1.append(i*2)

@pytest.fixture()
def data_featurization():
    smiles, target, features = load_organic_density()
    
    # split 0.9 train / 0.1 test
    df = pd.concat([smiles, target], axis=1)
    
    return df[:10]

@pytest.fixture()
def data_without_featurization():
    smiles, target, features = load_organic_density()
    
    df = pd.concat([features, target], axis=1)
    return df[:10]


def test_screener_types(data_featurization, data_without_featurization):
    
    df = data_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=True, smiles="smiles", n_gen=2, screener_type="regressor", output_file="testing.txt")
    scores = MS.screen_models(n_best=4)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4

    df = data_without_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=False, smiles=None, screener_type="regressor", n_gen=2, output_file="testing_without.txt")
    scores = MS.screen_models(n_best=4)

    assert len(scores) == 4

    with pytest.raises(ValueError):
        MS = ModelScreener(df=df, target="deny", featurization=False, smiles=None, screener_type="classifier", output_file="testing_without.txt")
        scores = MS.screen_models(n_best=4)






