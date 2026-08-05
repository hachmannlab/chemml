import os

import pytest
import pandas as pd
import json
# import numpy as np
from chemml.autoML import ModelScreener
# from chemml.chem import Molecule
from chemml.datasets import load_organic_density

# x1=[]
# y1=[]
# for i in range(0, 10):
#     x1.append(i)
#     y1.append(i*2)

@pytest.fixture()
def data_featurization():
    smiles, target, _ = load_organic_density()
    
    # split 0.9 train / 0.1 test
    df = pd.concat([smiles, target], axis=1)
    
    return df[:50]

@pytest.fixture()
def data_without_featurization():
    _, target, features = load_organic_density()
    
    df = pd.concat([features, target], axis=1)
    return df[:50]


def test_screener_types(data_featurization, data_without_featurization):
    
    df = data_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=True, smiles="smiles", n_gen=2, screener_type="regressor", output_dir="testing")
    scores = MS.screen_models(n_best=4)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4

    df = data_without_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=False, smiles=None, screener_type="regressor", n_gen=2, output_dir="testing_without")
    scores = MS.screen_models(n_best=4)

    assert len(scores) == 4

    with pytest.raises(ValueError):
        MS = ModelScreener(df=df, target="deny", featurization=False, smiles=None, screener_type="classifier", output_dir="testing_without")
        scores = MS.screen_models(n_best=4)


def test_screener_types_multicore(data_featurization, data_without_featurization):
    
    df = data_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=True, smiles="smiles", n_gen=2, screener_type="regressor", output_dir="testing")
    scores = MS.screen_models(n_best=4, multi_core=True)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4

    df = data_without_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=False, smiles=None, screener_type="regressor", n_gen=2, output_dir="testing_without")
    scores = MS.screen_models(n_best=4, multi_core=True)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4


def test_export_best_model_bundle(data_without_featurization, tmp_path):
    df = data_without_featurization
    export_dir = os.path.join(tmp_path, "best_model_bundle")

    MS = ModelScreener(
        df=df,
        target="density_Kg/m3",
        featurization=False,
        smiles=None,
        screener_type="regressor",
        n_gen=1,
        output_dir=export_dir
    )

    scores = MS.screen_models(n_best=2)
    assert len(scores) == 2

    best_model_path = os.path.join(export_dir, "best_model")

    metadata_path = os.path.join(best_model_path, "metadata.json")
    model_artifact_path = os.path.join(best_model_path, "model.pkl")
    feature_order_path = os.path.join(best_model_path, "feature_order.json")
    scaler_path = os.path.join(best_model_path, "scaler.pkl")

    assert os.path.exists(metadata_path)
    assert os.path.exists(model_artifact_path)
    assert os.path.exists(feature_order_path)
    assert os.path.exists(scaler_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert metadata["model_name"] == scores.iloc[0]["Model"]
    assert metadata["feature_key"] == scores.iloc[0]["Feature"]
    assert metadata["run_key"] == scores.iloc[0]["run_key"]

    with open(feature_order_path, "r") as f:
        feature_order = json.load(f)

    assert isinstance(feature_order, list)
    assert len(feature_order) == MS.x_list[metadata["feature_key"]].shape[1]
