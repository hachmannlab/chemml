import pytest
from chemml.published.RI import LorentzLorenz


@pytest.fixture()
def data():
    from chemml.published.RI import load_small_organic_data_10k
    _, targets = load_small_organic_data_10k()
    
    import numpy as np
    import pandas as pd
    
    features = [pd.DataFrame(np.random.random((50, 100))) for _ in range(4)]
    
    return features, targets
    


def test_feature_sizes(data):
    
    features, t = data
    targets = [t[t.columns[0]], t[t.columns[1]], t[t.columns[2]]]
    
    # vary input features
    for n in range(2, 5):
        ll_model = LorentzLorenz(n_features=n)
        X_train, X_test, y_train, y_test, scaler_y = ll_model.preprocessing(features=features[4-n:], targets=targets, return_scaler=True)
        ll_model = ll_model.fit(X_train, y_train[:50])
        y_pred = ll_model.predict(X_test)



def test_exceptions(data):
    features, t = data
    targets = [t[t.columns[0]], t[t.columns[1]], t[t.columns[2]]]
    
    # target sizes
    ll_model = LorentzLorenz(n_features=2)
    with pytest.raises(ValueError) as e:
        X_train, X_test, y_train, y_test, scaler_y = ll_model.preprocessing(features=features[2:], targets=targets[:-1], return_scaler=True)
        
    error_msg = e.value
    assert 'length of the target list should be 3' in str(error_msg)






