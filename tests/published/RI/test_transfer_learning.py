import pytest
from chemml.published.RI import load_model, load_small_organic_data_10k
from chemml.models import MLP, TransferLearning
import tensorflow as tf


@pytest.fixture()
def data():
    _, targets = load_small_organic_data_10k()
    
    import numpy as np
    import pandas as pd
    
    Xtrain, Xtest = [pd.DataFrame(np.random.random((50, 1893))) for _ in range(2)]
    Wtrain, Wtest = [pd.DataFrame(np.random.random((50, 100))) for _ in range(2)]
    
    return Xtrain.values, Xtest.values, Wtrain.values, Wtest.values, targets['polarizability'].values.reshape(-1,1)



def test_tl(data):

    ################### CHILD MODEL ###################
    # initialize a ChemML MLP object
    tzvp_model = MLP(learning_rate=0.01, batch_size=200, nepochs=5)
    
    # define layers
    tzvp_model.layers=[('Dense', {'units': 8,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(0.001),
                                    'activation': 'relu'
                                }),
                        ('Dense', {
                                'units': 1,
                                'activation': 'linear'
                        })
                        ]
    
    ################### PARENT MODEL ###################
    # initialize a ChemML MLP object
    mlp = MLP()
    
    # load pre-trained parent model
    svp_model = mlp.load(load_model('dragon', 'polarizability'))
    
    # initialize a TransferLearning object
    tl = TransferLearning(base_model=svp_model)
    
    Xtrain, Xtest, Wtrain, Wtest, targets = data
    
    # transfer the hidden layers from parent model to child model and fit the model to the new data
    combined_model = tl.transfer(Xtrain, targets[:50], tzvp_model)

    # predictions on test set
    y_pred = combined_model.predict(Xtest)
    
    
    
    
    #### TEST INCORRECT FEATURE SIZE ####
    with pytest.raises(ValueError) as e:
        combined_model = tl.transfer(Wtrain, targets[:50], tzvp_model)
        
    error_msg = e.value
    assert 'No. of Features for new model should be the same as that of the base model' in str(error_msg)







