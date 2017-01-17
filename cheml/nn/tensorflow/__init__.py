"""
The 'cheml.nn' module includes neural_net, 
last modified date: April 24, 2016
"""

from .mlp import mlp_classification
from .mlp import mlp_regression
# from .nn_SGD_Distributed import nn_dsgd_train, nn_dsgd_output
#Zfrom .nn_SGD_tensorflow import nn_tf
#from .nn_SGD_theano import nn_theano



__all__ = [
    'mlp_classification',
    'mlp_regression',
    'convolutional',
    '',
]
