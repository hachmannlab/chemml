"""
Generate features vectors for atoms and bonds

# Source
This code is adapted from:
    - https://github.com/HIPS/neural-fingerprint/blob/2e8ef09/neuralfingerprint/features.py
    - https://github.com/HIPS/neural-fingerprint/blob/2e8ef09/neuralfingerprint/util.py
    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/preprocessing.py

# Copyright
This code is governed by the MIT licence:
    - https://github.com/HIPS/neural-fingerprint/blob/2e8ef09/license.txt
    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/license.txt

"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import deserialize as layer_from_config
from chemml.utils.utilities import mol_shapes_to_dims

def neighbour_lookup(atoms, edges, maskvalue=0, include_self=False):
    ''' Looks up the features of an all atoms neighbours, for a batch of molecules.

    Parameters
    __________
    
    atoms (K.tensor): of shape (batch_n, max_atoms, num_atom_features)
    edges (K.tensor): of shape (batch_n, max_atoms, max_degree) with neighbour
        indices and -1 as padding value
    maskvalue (numerical): the maskingvalue that should be used for empty atoms
        or atoms that have no neighbours (does not affect the input maskvalue
        which should always be -1!)
    include_self (bool): if True, the featurevector of each atom will be added
        to the list feature vectors of its neighbours

    Returns
    _______
    
    neigbour_features (K.tensor): of shape (batch_n, max_atoms(+1), max_degree,
        num_atom_features) depending on the value of include_self

    '''

    # The lookup masking trick: We add 1 to all indices, converting the
    #   masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    # We then add a padding vector at index 0 by padding to the left of the
    #   lookup matrix with the value that the new mask should get
    # Theano padding:
    # masked_atoms = temporal_padding(atoms, (1,0), padvalue=maskvalue)
    # Tensorflow padding:
    paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
    masked_atoms = tf.pad(atoms, paddings, "CONSTANT")

    # Import dimensions
    atoms_shape = K.shape(masked_atoms)
    batch_n = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]

    edges_shape = K.shape(masked_edges)
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (batch_n, 1, 1)
    # Theano arange and cast:
    # offset = K.reshape(T.arange(batch_n, dtype=K.dtype(masked_edges)), offset_shape)
    # offset *= lookup_size
    # Tensorflow range and backend cast:
    offset = K.reshape(K.cast(tf.range(batch_n, dtype='int32'), dtype=K.dtype(masked_edges)), offset_shape)
    offset *= K.cast(lookup_size, dtype=K.dtype(offset))


    # apply offset to account for the fact that after reshape, all individual
    #   batch_n indices will be combined into a single big index
    flattened_atoms = K.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = K.reshape(masked_edges + offset, (batch_n, -1))

    # Gather flattened
    # flattened_result = K.gather(flattened_atoms, flattened_edges)
    # Tensorflow/backend cast:
    flattened_result = K.gather(flattened_atoms, K.cast(flattened_edges, dtype='int32'))

    # Unflatten result
    output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
    #output = T.reshape(flattened_result, output_shape)
    # Tensorflow/backend reshape:
    output = K.reshape(flattened_result, output_shape)

    if include_self:
        # return K.concatenate([K.expand_dims(atoms, dim=2), output], axis=2)
        # Tensorflow concat:
        return tf.concat([K.expand_dims(atoms, axis=2), output], axis=2) # Keras 2: raplaced dim with axis
    return output
    
class NeuralGraphHidden(layers.Layer):
    ''' 
    Hidden Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
        2015). This layer takes a graph as an input. The graph is represented as by
        three tensors.
        - The atoms tensor represents the features of the nodes.
        - The bonds tensor represents the features of the edges.
        - The edges tensor represents the connectivity (which atoms are connected to
            which)
        It returns the convolved features tensor, which is very similar to the atoms
        tensor. Instead of each node being represented by a num_atom_features-sized
        vector, each node now is represented by a convolved feature vector of size
        conv_width.
        # Example
            Define the input:
            ```python
                atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
                bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
                edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
            ```
            The `NeuralGraphHidden` can be initialised in three ways:
            1. Using an integer `conv_width` and possible kwags (`Dense` layer is used)
                ```python
                atoms1 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms0, bonds, edges])
                ```
            2. Using an initialised `Dense` layer
                ```python
                atoms1 = NeuralGraphHidden(Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
                ```
            3. Using a function that returns an initialised `Dense` layer
                ```python
                atoms1 = NeuralGraphHidden(lambda: Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
                ```
            Use `NeuralGraphOutput` to convert atom layer to fingerprint
        # Arguments
            inner_layer_arg: Either:
                1. an int defining the `conv_width`, with optional kwargs for the
                    inner Dense layer
                2. An initialised but not build (`Dense`) keras layer (like a wrapper)
                3. A function that returns an initialised keras layer.
            kwargs: For initialisation 1. you can pass `Dense` layer kwargs
        # Input shape
            List of Atom and edge tensors of shape:
            `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
              bond_features), (samples, max_atoms, max_degrees)]`
            where degrees referes to number of neighbours
        # Output shape
            New atom featuers of shape
            `(samples, max_atoms, conv_width)`
        # References
            - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
        '''
    
    def __init__(self, inner_layer_arg, **kwargs):
        # Initialise based on one of the three initialisation methods

        # Case 1: Check if inner_layer_arg is conv_width
        if isinstance(inner_layer_arg, (int)):
            self.conv_width = inner_layer_arg
            # Keras2: we assume all the kwargs should be passed to the Dense layer
            # dense_layer_kwargs, kwargs = filter_func_args(layers.Dense.__init__,
            # kwargs, overrule_args=['name'])
            self.create_inner_layer_fn = lambda: layers.Dense(self.conv_width, **kwargs) #dense_layer_kwargs)

        # Case 2: Check if an initialised keras layer is given
        elif isinstance(inner_layer_arg, layers.Layer):
            assert inner_layer_arg.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.conv_width = inner_layer_arg.compute_output_shape((None, 1))
            # layer_from_config will mutate the config dict, therefore create a get fn
            self.create_inner_layer_fn = lambda: layer_from_config(dict(
                                                    class_name=inner_layer_arg.__class__.__name__,
                                                    config=inner_layer_arg.get_config()))

        # Case 3: Check if a function is provided that returns a initialised keras layer
        elif callable(inner_layer_arg):
            example_instance = inner_layer_arg()
            assert isinstance(example_instance, layers.Layer), 'When initialising with a function, the function has to return a keras layer'
            assert example_instance.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.conv_width = example_instance.compute_output_shape((None, 1))
            self.create_inner_layer_fn = inner_layer_arg

        else:
            raise ValueError('NeuralGraphHidden has to be initialised with 1). int conv_widht, 2). a keras layer instance, or 3). a function returning a keras layer instance.')

        super(NeuralGraphHidden, self).__init__() # Keras2: all the kwargs will be passed to the Dense layer only

    def build(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        self.max_degree = max_degree

        # Add the dense layers (that contain trainable params)
        #   (for each degree we convolve with a different weight matrix)
        self.trainable_weights.append(self.trainable_weights)
        self.inner_3D_layers = []
        for degree in range(max_degree):

            # Initialise inner layer, and rename it
            inner_layer = self.create_inner_layer_fn()
            inner_layer_type = inner_layer.__class__.__name__.lower()
            inner_layer._name = self.name + '_inner_' + inner_layer_type + '_' + str(degree)

            # Initialise TimeDistributed layer wrapper in order to parallelise
            #   dense layer across atoms (3D)
            inner_3D_layer_name = self.name + '_inner_timedistributed_' + str(degree)
            inner_3D_layer = layers.TimeDistributed(inner_layer, name=inner_3D_layer_name)

            # Build the TimeDistributed layer (which will build the Dense layer)
            inner_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))

            # Store inner_3D_layer and it's weights
            self.inner_3D_layers.append(inner_3D_layer)
            self._trainable_weights += inner_3D_layer.trainable_weights

    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        # Import dimensions
        num_samples = atoms.shape[0]
        max_atoms = atoms.shape[1]
        num_atom_features = atoms.shape[-1]
        num_bond_features = bonds.shape[-1]

        # Create a matrix that stores for each atom, the degree it is
        # atom_degrees = K.sum(K.not_equal(edges, -1), axis=-1, keepdims=True)
        # backend cast to floatx:
        atom_degrees = K.sum(K.cast(K.not_equal(edges, -1), 'float64'), axis=-1, keepdims=True)

        # For each atom, look up the features of it's neighbour
        
        neighbour_atom_features = neighbour_lookup(atoms, edges, include_self=True)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = K.sum(neighbour_atom_features, axis=-2)

        # Sum the edge features for each atom
        summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        # summed_features = K.concatenate([summed_atom_features, summed_bond_features], axis=-1)
        # Tensorflow concat:
        summed_features = tf.concat([summed_atom_features, summed_bond_features], axis=-1)

        # For each degree we convolve with a different weight matrix
        new_features_by_degree = []
        for degree in range(self.max_degree):

            # Create mask for this degree
            atom_masks_this_degree = K.cast(K.equal(atom_degrees, degree), K.floatx())

            # Multiply with hidden merge layer
            #   (use time Distributed because we are dealing with 2D input/3D for batches)
            # Add keras shape to let keras now the dimensions
            summed_features._keras_shape = (None, max_atoms, num_atom_features+num_bond_features)
            new_unmasked_features = self.inner_3D_layers[degree](summed_features)

            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features_by_degree.append(new_masked_features)

        # Finally sum the features of all atoms
        new_features = layers.add(new_features_by_degree)

        return new_features

    def compute_output_shape(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        return (num_samples, max_atoms, self.conv_width)

    @classmethod
    def from_config(cls, config):
        # Use layer build function to initialise new NeuralHiddenLayer
        inner_layer_config = config.pop('inner_layer_config')
        # create_inner_layer_fn = lambda: layer_from_config(inner_layer_config.copy())
        def create_inner_layer_fn():
            return layer_from_config(deepcopy(inner_layer_config))

        layer = cls(create_inner_layer_fn, **config)
        return layer

    def get_config(self):
        config = super(NeuralGraphHidden, self).get_config()

        # Store config of (a) inner layer of the 3D wrapper
        inner_layer = self.inner_3D_layers[0].layer
        config['inner_layer_config'] = dict(config=inner_layer.get_config(),
                                            class_name=inner_layer.__class__.__name__)
        return config
    
class NeuralGraphOutput(layers.Layer):
    ''' Output Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
    2015). This layer takes a graph as an input. The graph is represented as by
    three tensors.

    - The atoms tensor represents the features of the nodes.
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)

    It returns the fingerprint vector for each sample for the given layer.

    According to the original paper, the fingerprint outputs of each hidden layer
    need to be summed in the end to come up with the final fingerprint.

    # Example
        Define the input:
        ```python
            atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
            bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
            edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
        ```

        The `NeuralGraphOutput` can be initialised in three ways:
        1. Using an integer `fp_length` and possible kwags (`Dense` layer is used)
            ```python
            fp_out = NeuralGraphOutput(fp_length, activation='relu', bias=False)([atoms0, bonds, edges])
            ```
        2. Using an initialised `Dense` layer
            ```python
            fp_out = NeuralGraphOutput(Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        3. Using a function that returns an initialised `Dense` layer
            ```python
            fp_out = NeuralGraphOutput(lambda: Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
            ```

        Predict for regression:
        ```python
        main_prediction = Dense(1, activation='linear', name='main_prediction')(fp_out)
        ```

    # Arguments
        inner_layer_arg: Either:
            1. an int defining the `fp_length`, with optional kwargs for the
                inner Dense layer
            2. An initialised but not build (`Dense`) keras layer (like a wrapper)
            3. A function that returns an initialised keras layer.
        kwargs: For initialisation 1. you can pass `Dense` layer kwargs

    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours

    # Output shape
        Fingerprints matrix
        `(samples, fp_length)`

    # References
        - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)

    '''

    def __init__(self, inner_layer_arg, **kwargs):
        # Initialise based on one of the three initialisation methods

        # Case 1: Check if inner_layer_arg is fp_length
        if isinstance(inner_layer_arg, (int)):
            self.fp_length = inner_layer_arg
            # we assume all the kwargs should be passed to the Dense layer
            # dense_layer_kwargs, kwargs = filter_func_args(layers.Dense.__init__,
            # kwargs, overrule_args=['name'])
            self.create_inner_layer_fn = lambda: layers.Dense(self.fp_length, **kwargs) # dense_layer_kwargs)

        # Case 2: Check if an initialised keras layer is given
        elif isinstance(inner_layer_arg, layers.Layer):
            assert inner_layer_arg.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.fp_length = inner_layer_arg.compute_output_shape((None, 1))
            self.create_inner_layer_fn = lambda: inner_layer_arg

        # Case 3: Check if a function is provided that returns a initialised keras layer
        elif callable(inner_layer_arg):
            example_instance = inner_layer_arg()
            assert isinstance(example_instance, layers.Layer), 'When initialising with a function, the function has to return a keras layer'
            assert example_instance.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.fp_length = example_instance.compute_output_shape((None, 1))
            self.create_inner_layer_fn = inner_layer_arg

        else:
            raise ValueError('NeuralGraphHidden has to be initialised with 1). int conv_widht, 2). a keras layer instance, or 3). a function returning a keras layer instance.')

        super(NeuralGraphOutput, self).__init__() # Keras2: all the kwargs will be passed to the Dense layer only

    def build(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        # Add the dense layer that contains the trainable parameters
        # Initialise dense layer with specified params (kwargs) and name
        inner_layer = self.create_inner_layer_fn()
        inner_layer_type = inner_layer.__class__.__name__.lower()
        inner_layer._name = self.name + '_inner_'+ inner_layer_type

        # Initialise TimeDistributed layer wrapper in order to parallelise
        #   dense layer across atoms
        inner_3D_layer_name = self.name + '_inner_timedistributed'
        self.inner_3D_layer = layers.TimeDistributed(inner_layer, name=inner_3D_layer_name)

        # Build the TimeDistributed layer (which will build the Dense layer)
        self.inner_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))

        # Store dense_3D_layer and it's weights
        self._trainable_weights = self.inner_3D_layer.trainable_weights


    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        # Import dimensions
        num_samples = atoms.shape[0]
        max_atoms = atoms.shape[1]
        num_atom_features = atoms.shape[-1]
        num_bond_features = bonds.shape[-1]

        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        # atom_degrees = K.sum(K.not_equal(edges, -1), axis=-1, keepdims=True)
        # backend cast to floatx:
        atom_degrees = K.sum(K.cast(K.not_equal(edges, -1), 'float64'), axis=-1, keepdims=True)
        general_atom_mask = K.cast(K.not_equal(atom_degrees, 0), K.floatx())

        # Sum the edge features for each atom
        summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        # atoms_bonds_features = K.concatenate([atoms, summed_bond_features], axis=-1)
        # Tensorflow concat:
        atoms_bonds_features = tf.concat([atoms, summed_bond_features], axis=-1)

        # Compute fingerprint
        atoms_bonds_features._keras_shape = (None, max_atoms, num_atom_features+num_bond_features)
        fingerprint_out_unmasked = self.inner_3D_layer(atoms_bonds_features)

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        # Sum across all atoms
        final_fp_out = K.sum(fingerprint_out_masked, axis=-2)

        return final_fp_out

    def compute_output_shape(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        return (num_samples, self.fp_length)

    @classmethod
    def from_config(cls, config):
        # Use layer build function to initialise new NeuralGraphOutput
        inner_layer_config = config.pop('inner_layer_config')
        create_inner_layer_fn = lambda: layer_from_config(deepcopy(inner_layer_config))

        layer = cls(create_inner_layer_fn, **config)
        return layer

    def get_config(self):
        config = super(NeuralGraphOutput, self).get_config()

        # Store config of inner layer of the 3D wrapper
        inner_layer = self.inner_3D_layer.layer
        config['inner_layer_config'] = dict(config=inner_layer.get_config(),
                                            class_name=inner_layer.__class__.__name__)
        return config

class NeuralGraphPool(layers.Layer):
    ''' Pooling layer in a Neural graph, for each atom, takes the max for each
    feature between the atom and it's neighbours

    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours

    # Output shape
        New atom features (of same shape:)
        `(samples, max_atoms, atom_features)`
    '''
    def __init__(self, **kwargs):
        super(NeuralGraphPool, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        # For each atom, look up the featues of it's neighbour
        neighbour_atom_features = neighbour_lookup(atoms, edges, maskvalue=-inf,
                                                   include_self=True)

        # Take max along `degree` axis (2) to get max of neighbours and self
        max_features = K.max(neighbour_atom_features, axis=2)

        # atom_degrees = K.sum(K.not_equal(edges, -1), axis=-1, keepdims=True)
        # backend cast to floatx:
        atom_degrees = K.sum(K.cast(K.not_equal(edges, -1), K.floatx()), axis=-1, keepdims=True)
        general_atom_mask = K.cast(K.not_equal(atom_degrees, 0), K.floatx())

        return max_features * general_atom_mask

    def compute_output_shape(self, inputs_shape):

        # Only returns `atoms` tensor
        return inputs_shape[0]

class AtomwiseDropout(layers.Layer):
    ''' Performs dropout over an atom feature vector where each atom will get
    the same dropout vector.

    Eg. With an input of `(batch_n, max_atoms, atom_features)`, a dropout mask of
    `(batch_n, atom_features)` will be generated, and repeated `max_atoms` times

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    '''
    def __init__(self, p, **kwargs):
        self.dropout_layer = layers.Dropout(p)
        self.uses_learning_phase = self.dropout_layer.uses_learning_phase
        self.supports_masking = True
        super(AtomwiseDropout, self).__init__(**kwargs)

    def _get_noise_shape(self, x):
        return None

    def call(self, inputs, mask=None):
        # Import (symbolic) dimensions
        max_atoms = K.shape(inputs)[1]

        # By [farizrahman4u](https://github.com/fchollet/keras/issues/3995)
        ones = layers.Lambda(lambda x: (x * 0 + 1)[:, 0, :], output_shape=lambda s: (s[0], s[2]))(inputs)
        dropped = self.dropout_layer(ones)
        dropped = layers.RepeatVector(max_atoms)(dropped)
        return layers.Lambda(lambda x: x[0] * x[1], output_shape=lambda s: s[0])([inputs, dropped])

    def get_config(self):
        config = super(AtomwiseDropout, self).get_config()
        config['p'] = self.dropout_layer.p
        return config

#TODO: Add GraphWiseDropout layer, that creates masks for each degree separately.
