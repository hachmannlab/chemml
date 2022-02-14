import pytest


def test_load_hyperparameters():
    
    from chemml.published.RI import load_hyperparameters
    from chemml.models import MLP
    
    for f in ['dragon', 'morgan', 'hap', 'htt']:
        for t in ['polarizability', 'refractive_index', 'number_density']:
            a = MLP()
            a_with_hyp = load_hyperparameters(a, f, t, 'single')
            # print(vars(a_with_hyp))
    


def test_load_small_organic_data():

    from chemml.published.RI import load_small_organic_data
    m, t = load_small_organic_data()
    
    assert len(m) == 100000
    assert t.shape == (100000, 3)


def test_load_small_organic_data_10k():

    from chemml.published.RI import load_small_organic_data_10k
    m, t = load_small_organic_data_10k()
    
    assert len(m) == 10000
    assert t.shape == (10000, 3)


def test_load_model():
    from chemml.published.RI import load_model
    from chemml.models import MLP
    
    # single DNNs
    for f in ['dragon', 'morgan', 'hap', 'htt']:
        for t in ['polarizability', 'refractive_index', 'number_density']:
            model_path = load_model(f, t, model='single')
            mlp = MLP()
            mlp = mlp.load(model_path)
            
    # physics infused
    model_path = load_model(f, t, model='lorentz_lorenz')
    
    # to avoid confusion in the two load_model functions from chemml and keras, import one only after the other has been used completely.
    from tensorflow.keras.models import load_model
    
    mlp = load_model(model_path)
    
    






















































