from tensorflow import keras
from tensorflow.keras.layers import Input
from chemml.models.mlp import MLP
from tensorflow.keras.models import Model
import chemml
import torch
from torch import nn
import torch.nn.functional as F
from importlib import import_module



class TransferLearning(object):
    """
    Class used to facilitate transfer learning from a parent (or head) model to a child model.
    Freezes the layers (weights and biases) of the pre-trained base 
    model and removes the output layers and appends the child model to the base model.
    
    Parameters
    __________
    base_model: chemml.models.MLP object, tensorflow.python.keras object, or a torch.nn.Module object
        pre-trained base model
    
    n_features: int or None (default=None)
        no. of input features provided for training the base model. 
        If base_model is a tensorflow.python.keras object or torch.nn.Modules object, then n_features must be int, else it 
        can be None.
    
    n_layers: int, optional (default=None)
            remove the last 'n' hidden layers from the base model. 
            Note that this number should not include the output layer.
    
    """
    def __init__(self, base_model, nfeatures=None, n_layers=None):
        # getting the tensorflow or pytorch model from the chemml object
        self.base_model = base_model
        self.nfeatures = nfeatures
        
        if n_layers is not None:
            if not isinstance(n_layers, int):
                raise ValueError('n_layers should be an integer or None.')
            else:
                self.n_layers= n_layers
        else:
            self.n_layers = 0

        if isinstance(self.base_model, MLP):
            self.derived_model = True
            self.nfeatures = self.base_model.nfeatures
            self.base_model = self.base_model.get_model(include_output=False, n_layers=self.n_layers)
            self.engine = base_model.engine

        elif 'tensorflow.python.keras.engine.sequential' in str(type(self.base_model)):
            self._init_tensorflow()
        
        elif 'torch.nn.modules.container.Sequential' in str(type(self.base_model)):
            self._init_pytorch()

        else:
            raise ValueError('Base model has to be a chemml.models.MLP object, tensorflow.python.keras object or a torch.nn.Modules object')


    def _init_tensorflow(self):
        self.engine = 'tensorflow'
        if self.nfeatures == None:
            raise ValueError('When using a tensorflow or pytorch model as a base model, the no. of features for base model have to be provided.')
        # removing output from base tensorflow base model
        if not self.derived_model:
            for _ in range(self.n_layers+1):
                self.base_model.pop()
        self.base_model.trainable = False


    def _init_pytorch(self):
        self.engine = 'pytorch'
        if self.nfeatures == None:
            raise ValueError('When using a tensorflow or pytorch model as a base model, the no. of features for base model have to be provided.')
        # removing output from base pytorch base model
        if not self.derived_model:
            self.base_model =  self.base_model[:-(self.n_layers)]
        for i in self.base_model:
            for params in i.parameters():
                params.requires_grad = False


    def transfer(self, X, y, child_model):
        """
        Adds the base model's frozen layers (without its input and output layers) 
        to the child model and fits the new model to the training data. 
        
        Parameters
        __________
        X, y: numpy.ndarray
            X is an array of features and y is an array of the target values
            X should have the same input features (columns) as the base model.
        
        child_model: chemml.models.mlp.MLP object
            chemml model created with all the parameters and options required for 
            the final transfer learned model
        
        Returns
        _______
        child_model: chemml.models.mlp.MLP object
            The trained transfer-learned child model.
            
        """
        if not type(child_model) is chemml.models.mlp.MLP:
            raise ValueError('The child model should be a chemml.models.MLP object')   

        if not (self.nfeatures == X.shape[1] and child_model.nfeatures == self.nfeatures):
            if self.nfeatures==(X.shape[1],) and self.engine=='tensorflow':
               pass
            else: 
                raise ValueError('No. of Features for new model should be the same as that of the base model')
        
        if not type(child_model.model) == type(self.base_model):
            raise TypeError('The underlying engine for the child model should be the same as the base_model')
            
        # tensorflow
        if self.engine == 'tensorflow':
            new_child_model = keras.Sequential()
            keras_layer_module = import_module('tensorflow.keras.layers')
            layer = getattr(keras_layer_module, child_model.layers[1][0])
            new_child_model.add(layer(**child_model.layers[1][1]))

            for i in child_model.model.layers[1:]:
                new_child_model.add(i)
            model = keras.Sequential([self.base_model, new_child_model])
            model.compile(loss=child_model.loss, optimizer=child_model.opt)
            model_layers = list(model.layers)
        else:
            # pytorch
            new_layer = nn.Linear(in_features = self.base_model[-1].in_features, out_features = child_model.model[2].in_features)
            new_layer.state_dict()['weight'] = self.base_model[-1].state_dict()['weight']
            new_layer.state_dict()['bias'] = self.base_model[-1].state_dict()['bias']
            
            layers = list(self.base_model.children())[:-2]+ list(nn.Sequential(new_layer,self.base_model[-2]).children()) + list(child_model.model.children())[1:]
            print([c for c in nn.Sequential(new_layer,self.base_model[-1]).children()])
            model = nn.Sequential(*layers)
            
            model_layers = list(model.children())

        new_chemml_model = MLP(engine=self.engine, nfeatures=self.nfeatures, learning_rate=child_model.learning_rate, 
                                nepochs= child_model.nepochs, batch_size=child_model.batch_size, alpha=child_model.alpha, 
                                loss=child_model.loss, opt_config=child_model.opt_config, layer_config_file=model_layers)
        print(new_chemml_model.model)
        new_chemml_model.fit(X, y)

        return new_chemml_model

