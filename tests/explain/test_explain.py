# create pytest fixture which returns fitted pytorch model (engine_model in eval state), Xte (to use an instance to create local explanations), and feature column names


import pytest

import pandas as pd
import shap
from chemml.models import MLP
from chemml.datasets import load_organic_density

from sklearn.preprocessing import StandardScaler
from chemml.explain import Explain

@pytest.fixture()
def data():
    _, y, X = load_organic_density()
    columns = list(X.columns)
    y = y.values.reshape(y.shape[0], 1).astype('float32')
    X = X.values.reshape(X.shape[0], X.shape[1]).astype('float32')

    # split 0.9 train / 0.1 test
    ytr = y[:450, :]
    yte = y[450:, :]
    Xtr = X[:450, :]
    Xte = X[450:, :]
    
    scale = StandardScaler()
    scale_y = StandardScaler()
    Xtr = scale.fit_transform(Xtr)
    Xte = scale.transform(Xte)
    ytr = scale_y.fit_transform(ytr)

    # PYTORCH
    r1_pytorch = MLP(engine='pytorch',nfeatures=Xtr.shape[1], nneurons=[100,100,100], activations=['ReLU','ReLU','ReLU'],
            learning_rate=0.001, alpha=0.0001, nepochs=100, batch_size=100, loss='mean_squared_error', 
            regression=True, nclasses=None, layer_config_file=None, opt_config='Adam')

    r1_pytorch.fit(Xtr, ytr)
    engine_model = r1_pytorch.get_model()
    engine_model.eval()


    return Xtr, Xte, engine_model, columns


def test_deepshap(data):

    Xtr, Xte, engine_model, columns = data
    X_instance = Xtr[0]
    exp = Explain(X_instance, engine_model, columns)
    # X[np.random.choice(X.shape[0], 10, replace=False)]

    explanation, shap_obj = exp.DeepSHAP(Xtr[1:10])
    
    assert len(explanation.columns) == len(columns)
    assert len(explanation) == X_instance.shape[0] or len(explanation) == X_instance.ndim
    assert isinstance(shap_obj, shap.DeepExplainer) == True

def test_lrp(data):
    Xtr, Xte, engine_model, columns = data
    X_instance = Xtr[0]
    exp = Explain(X_instance, engine_model, columns)
    print(type(Xtr[1:10]))

    # strategies + local relevance
    for strategy in ['zero','eps','composite']:
        explanation, gb = exp.LRP(strategy=strategy)

        assert isinstance(explanation,pd.DataFrame)
        assert len(explanation.columns) == len(columns)
        assert len(explanation) == X_instance.shape[0] or len(explanation) == X_instance.ndim
        assert gb is None

    # strategies + global relevance
    X_instance = Xte
    exp = Explain(X_instance, engine_model, columns)
    for strategy in ['zero','eps','composite']:

        explanation, gb = exp.LRP(strategy=strategy,global_relevance=True)

        assert isinstance(explanation,pd.DataFrame)
        assert len(explanation.columns) == len(columns)
        assert len(explanation) == X_instance.shape[0] or len(explanation) == X_instance.ndim
        assert isinstance(gb,pd.DataFrame)
        assert list(gb.columns) == ['Mean Absolute Relevance Score','Mean Relevance Score']
        assert len(gb) == len(columns)


def test_lime(data):
    Xtr, Xte, engine_model, columns = data
    for X_instance in [Xte[0], Xte[1:5]]:

        exp = Explain(X_instance, engine_model, columns)
        explanation = exp.LIME(Xtr)

        assert isinstance(explanation[0], pd.DataFrame) and isinstance(explanation, list) 
        assert len(explanation) == X_instance.shape[0] or len(explanation) == X_instance.ndim
