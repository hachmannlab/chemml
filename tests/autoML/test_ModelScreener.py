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
    
    return df[:100]

@pytest.fixture()
def data_without_featurization():
    _, target, features = load_organic_density()
    
    df = pd.concat([features, target], axis=1)
    return df[:100]


def test_screener_types(data_featurization, data_without_featurization):
    
    df = data_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=True, smiles="smiles", n_gen=2, screener_type="regressor", output_file="testing.txt")
    scores = MS.screen_models(n_best=4)
    

    #if scores is not empty, everything is okay
    assert len(scores) == 4

    # Testing multi-core screener
    scores = MS.screen_models(n_best=4, multi_core=True)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4

    df = data_without_featurization
    MS = ModelScreener(df=df, target="density_Kg/m3", featurization=False, smiles=None, screener_type="regressor", n_gen=2, output_file="testing_without.txt")
    scores = MS.screen_models(n_best=4)

    assert len(scores) == 4

    # Testing multi-core screener
    scores = MS.screen_models(n_best=4, multi_core=True)
    
    #if scores is not empty, everything is okay
    assert len(scores) == 4


    with pytest.raises(ValueError):
        MS = ModelScreener(df=df, target="deny", featurization=False, smiles=None, screener_type="classifier", output_file="testing_without.txt")
        scores = MS.screen_models(n_best=4)


def test_export_best_model_bundle(data_without_featurization, tmp_path):
    df = data_without_featurization
    export_dir = tmp_path / "best_model_bundle"

    MS = ModelScreener(
        df=df,
        target="density_Kg/m3",
        featurization=False,
        smiles=None,
        screener_type="regressor",
        n_gen=1,
        output_file=str(tmp_path / "screening_output.txt"),
    )

    scores = MS.screen_models(n_best=2, best_model_output_file=str(export_dir))
    assert len(scores) == 2

    metadata_path = export_dir / "metadata.json"
    model_artifact_path = export_dir / "model_artifact"
    feature_order_path = export_dir / "feature_order.json"
    scaler_path = export_dir / "scaler.pkl"

    assert metadata_path.exists()
    assert model_artifact_path.exists()
    assert feature_order_path.exists()
    assert scaler_path.exists()

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert metadata["model_name"] == scores.iloc[0]["Model"]
    assert metadata["feature_key"] == scores.iloc[0]["Feature"]
    assert metadata["run_key"] == scores.iloc[0]["run_key"]

    with open(feature_order_path, "r") as f:
        feature_order = json.load(f)

    assert isinstance(feature_order, list)
    assert len(feature_order) == MS.x_list[metadata["feature_key"]].shape[1]
