import chemml
from chemml.models import MLP 
import pandas as pd
import pkg_resources
import os


def load_hyperparameters(mlp_obj, features, targets, model_type):
    
    """ Function to return an mlp model with optimized hyperparameters.
    
    Parameters 
    ----------
    mlp_obj: <chemml.models.keras.mlp.MLP> object
        model object in which hyperparameters are to be loaded
        
    features: str
        feature set to be used -- "dragon", "hap", "htt", "morgan" 
    
    targets: str
        target property -- "refractive_index", "number_density", "polarizability"
        
    model_type: str
        options -- "single", "multitask", "lorentz_lorenz", "single_10k", "transfer_learning"
    """
    
    hyperparameters = hyperparams()
    layers_config = layers_conf()
    
    if type(mlp_obj) is not chemml.models.keras.mlp.MLP:
        raise TypeError("mlp_obj should be a <chemml.models.keras.mlp.MLP> object")
        
    if features not in ['dragon', 'morgan', 'hap', 'htt']:
        raise ValueError("feature should be either 'dragon', 'morgan', 'hap' or 'htt' ")
    
    if targets not in ['polarizability', 'refractive_index', 'number_density']:
        raise ValueError("targets should be either 'polarizability', 'refractive_index' or 'number_density' ")
    
    if model_type not in ['single', 'multitask', 'lorentz_lorenz', 'single_10k', 'transfer_learning']:
        raise ValueError("model_type should be either 'single', 'multitask', 'lorentz_lorenz', 'transfer_learning' or 'single_10k' ")
        
        
    mlp_obj = MLP(*hyperparameters[model_type][features][targets])
    mlp_obj.layers = layers_config[model_type][features][targets]
    
    return mlp_obj


def load_small_organic_data():
    df = pd.read_csv(pkg_resources.resource_filename('chemml', os.path.join('published', 'RI', 'input_files', 'properties.csv')))
    molecules = df['Mol_Smiles']
    
    sub_cols = ['RI_LL', 'Pol_geom', 'Den_MD']
    targets = df[sub_cols]
    targets.columns = ['refractive_index', 'polarizability', 'number_density']
    
    return molecules, targets


def load_small_organic_data_10k():
    df = pd.read_csv(pkg_resources.resource_filename('chemml', os.path.join('published', 'RI', 'input_files', 'properties_10k.csv')))
    molecules = df['Mol_Smiles']
    
    sub_cols = ['RI_LL', 'Pol_tz', 'Den_MD']
    # ['Pol_DFT', 'Pol_geom', 'Pol_tz']     ['RI_LL', 'RI_geom']
    targets = df[sub_cols]
    targets.columns = ['refractive_index', 'polarizability', 'number_density']
    
    return molecules, targets


def load_model(features, targets, model='single'):
    if model == 'single':
        model_path = pkg_resources.resource_filename('chemml', os.path.join('published', 'RI', 'trained_models', 'single', features + '_' + targets + '_chemml_model.csv'))
        
    if model == 'lorentz_lorenz':
        model_path = pkg_resources.resource_filename('chemml', os.path.join('published', 'RI', 'trained_models', 'lorentz_lorenz.h5'))
        
    if model == 'transfer_learning':
        model_path = pkg_resources.resource_filename('chemml', os.path.join('published', 'RI', 'trained_models', 'single', 'morgan' + '_' + 'polarizability' + '_chemml_model.csv'))
        
    return model_path






































def hyperparams():
    hyperparameters = {
                    "single": {
                        "dragon":  {
                                    "polarizability": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':1.69e-05}]], 
                                    "refractive_index": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':3.98e-05}]],
                                    "number_density": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':1.72e-05}]], 
                                },
                        "morgan": {
                                    "polarizability": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':8.78e-05}]], 
                                    "refractive_index": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':1.91e-05}]],
                                    "number_density": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':9.04e-05}]],
                        	    },
            	        "hap": {
                                    "polarizability": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':3.72e-05}]], 
                                    "refractive_index": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None,['Adam',{'learning_rate':2.59e-05}]],
                                    "number_density": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':1.83e-05}]],
                                },
                        "htt": {
                                    "polarizability": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':9.42e-05}]], 
                                    "refractive_index": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':2.95e-05}]],
                                    "number_density": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':1.88e-05}]],
                                } 
                            },
                    "multitask": {
                        
                            },
                    
                    "lorentz_lorenz": {
                        
                            },
                            
                    "single_10k": {
                        
                            },
                    "transfer_learning": {
                        "morgan": {
                                    "polarizability": [1, None, None, 0.01, 0.0, 500,50,'mean_squared_error', 'True', None, None, ['Adam',{'learning_rate':8.78e-05}]]
                        }
                        
                            }        
                    }
    return hyperparameters





def layers_conf():
    
    import tensorflow as tf
    
    layers_config ={"single": 
                {
                "dragon":  {
                            "polarizability": [
                                    ('Dense', {
                                    'units': 64 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(8.36e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.002}), 
                                    ('Dense', {
                                    'units': 32,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(8.36e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.002
                                    }), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(8.36e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.002
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                    ], 
                            "refractive_index": [
                                    ('Dense', {
                                    'units': 32 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(5.53e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0}), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(5.53e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0
                                    }), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(5.53e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(5.53e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(5.53e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                    ],
                            "number_density": [
                                    ('Dense', {
                                    'units': 128 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.22e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0}), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.22e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0
                                    }), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.22e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.22e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                            'units': 1,
                                            'activation': 'linear'    #do not change
                                    })
                                    ], 
                                
                            },
                "morgan": {
                            "polarizability": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.006}), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.006
                                    }), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.006
                                    }), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.006
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                ], 
                            "refractive_index": [
                                    ('Dense', {
                                    'units': 128 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(3.41e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.49}), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(3.41e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.49
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                    
                                 ],
                            "number_density": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.74e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.009
                                    }), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.74e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.009
                                    }), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.74e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.009
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                    ], 
                            },
                "hap": {
                            "polarizability": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(9.05e-4),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.12}), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(9.05e-4),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.12
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                 ]
                                , 
                            "refractive_index": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(3.12e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.047}), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(3.12e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.047}),
                                    ('Dense', {
                                    'units': 32,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(3.12e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.047}),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                 ] ,
                            "number_density": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(1.66e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.03}), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(1.66e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.03
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                 ],
                        },
                        
                "htt": {
                            "polarizability": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(4.54e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0}), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(4.54e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.0
                                    }), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(4.54e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(4.54e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(4.54e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.0
                                    }),
                                    ('Dense', {
                                            'units': 1,
                                            'activation': 'linear'    #do not change
                                    })
                                    ], 
                            "refractive_index": [
                                    ('Dense', {
                                    'units': 64 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.52e-4),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.019}), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.52e-4),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.019
                                    }), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(2.52e-4),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.019
                                    }),
                                    ('Dense', {
                                            'units': 1,
                                            'activation': 'linear'    #do not change
                                    })
                                    ],
                            "number_density": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(1.17e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.002}), 
                                    ('Dense', {
                                    'units': 32,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(1.17e-3),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.002
                                    }), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(1.17e-3),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.002
                                    }),
                                    ('Dense', {
                                            'units': 1,
                                            'activation': 'linear'    #do not change
                                    })
                                    ],
                        } 
                },
            "multitask":
                {
                    
                },
            "lorentz_lorenz":
                {
                    
                },
            "single_10k":
                {
                    
                },
            "transfer_learning": {
                "morgan": {
                            "polarizability": [
                                    ('Dense', {
                                    'units': 256 ,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.006}), 
                                    ('Dense', {
                                    'units': 64,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }), 
                                    ('Dropout', {
                                    'rate': 0.006
                                    }), 
                                    ('Dense', {
                                    'units': 128,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.006
                                    }), 
                                    ('Dense', {
                                    'units': 256,
                                    'kernel_initializer':tf.keras.initializers.glorot_uniform(seed=1369),
                                    'kernel_regularizer':tf.keras.regularizers.l2(7.26e-5),
                                    'activation': 'relu'
                                    }),
                                    ('Dropout', {
                                    'rate': 0.006
                                    }),
                                    ('Dense', {
                                    'units': 1,
                                    'activation': 'linear'    #do not change
                                    })
                                ],}
            }
            }
    return layers_config











