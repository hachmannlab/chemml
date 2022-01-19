from tensorflow.keras.layers import Input
from chemml.models.keras.mlp import MLP
from tensorflow.keras.models import Model
import chemml
from importlib import import_module

class TransferLearning(object):
    """
    Class used to facilitate transfer learning from a parent (or head) model to 
    a child model.Freezes the layers (weights and biases) of the pre-trained base 
    model and removes the input and output layers and adds it to the child model.
    
    Parameters
    __________
    base_model: chemml.models.keras.MLP object or tensorflow.python.keras object
        pre-trained base model
    
    n_features: int or None (default=None)
        no. of input features provided for training the base model. If base_model
        is a tensorflow.python.keras object then n_features must be int, else it 
        can be None.
    
    n_layers: int, optional (default=None)
            remove the last 'n' hidden layers from the base model. 
            Note that this number should not include the output layer.
    
    """
    def __init__(self, base_model, n_features=None, n_layers=None):
        # getting the keras model from the chemml object
        # self.base_model will now be a keras object without the initial layer
        if type(base_model) is chemml.models.keras.mlp.MLP:
            self.base_features = base_model.feature_size
            self.base_model = base_model.get_keras_model(include_output=False, n_layers=n_layers)
            
        elif 'tensorflow.python.keras' in str(type(base_model)):
            #removing output from base tensorflow model
            if n_layers is not None:
                if isinstance(n_layers, int):
                    self.base_model = Model(base_model.input, base_model.layers[-(2+n_layers)].output)
                else:
                    raise ValueError('n_layers should be an integer.')
            else:
                self.base_model = Model(base_model.input, base_model.layers[-2].output) 
            
            if n_features == None:
                raise ValueError('When using a keras model as a base model, the no. of features for base model have to be provided.')
            self.base_features = n_features
        else:
            raise ValueError('Base model has to be a chemml.models.MLP object or a tensorflow.python.keras object')
            
        self.base_model.trainable = False
        
    def transfer(self, X, y, child_model):
        """
        Adds the base model's frozen layers (without its input and output layers) 
        to the child model and fits the new model to the training data. 
        
        Parameters
        __________
        X, y: numpy.ndarray
            X is an array of features and y is an array of the target values
        
        child_model: chemml.models.keras.mlp.MLP object
            chemml model created with all the parameters and options required for 
            the final transfer learned model
        
        Returns
        _______
        child_model: chemml.models.keras.mlp.MLP object
            The trained transfer-learned child model.
            
        """
        
        if not self.base_features == X.shape[1]:
            raise ValueError('No. of Features for new model should be the same as that of the base model')
        
        if not type(child_model) is chemml.models.keras.mlp.MLP:
            raise ValueError('The child model should be a chemml.models.MLP object')
            
        input_layer = Input(X.shape[1],)
        # overriding the input layer of the base model with  the new input layer defined above
        x = self.base_model(input_layer, training=False)
        
        if len(child_model.layers) == 0:
            for i in range(child_model.nhidden):
                child_model.layers.append(('Dense', {
                    'units': child_model.nneurons[i],
                    'activation': child_model.activations[i]
                }))
            if child_model.is_regression:
                child_model.layers.append(('Dense', {
                    'units': 1,
                    'activation': 'linear'
                }))
            else:
                child_model.layers.append(('Dense', {
                    'units': self.nclasses,
                    'activation': 'softmax'
                }))
        
        # child_model.model.add(x)
        # layer_name, layer_params = child_model.layers[0]
        # layer_params['input_dim'] = X.shape[-1] #assigning input dimensions to the first layer in the sequential model
        keras_layer_module = import_module('tensorflow.keras.layers')
        for layer_name, layer_params in child_model.layers:
            layer = getattr(keras_layer_module, layer_name)
            layer = layer(**layer_params)(x)
            x = layer
            
        final_model = Model(input_layer, layer)
        final_model.compile(optimizer=child_model.opt,
                            loss=child_model.loss,
                            metrics=['mean_absolute_error'])
        
        
            # child_model.model.add(layer(**layer_params))
        
        child_model.model = final_model
        
        child_model.batch_size = X.shape[
            0] if X.shape[0] < child_model.batch_size else child_model.batch_size
        
        child_model.model.fit(
            x=X, y=y, epochs=child_model.nepochs, batch_size=child_model.batch_size)
        
        return child_model