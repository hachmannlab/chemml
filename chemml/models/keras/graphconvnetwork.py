from __future__ import division, print_function, absolute_import

# from keras.regularizers import l1_l2  # keras 1.2
from tensorflow.keras.regularizers import l1_l2  # keras 2.*
from tensorflow.keras.layers import Input, merge, Dense, Dropout, BatchNormalization
from tensorflow.keras import models

from .layers import NeuralGraphHidden, NeuralGraphOutput, NeuralGraphPool, AtomwiseDropout
from .utils import zip_mixed, is_iterable

def build_graph_conv_model(max_atoms, max_degree, num_atom_features, num_bond_features, learning_type, output_size=1, optimizer='adagrad', **kwargs):
	''' Builds and compiles a graph convolutional network with a regular neural
		network on top for regression.

	Especially usefull when using the sklearn `KerasClassifier` wrapper

	# Arguments
		max_atoms, max_degree, num_atom_features, num_bond_features (int): The
			dimensionalities used to create input layers.
		learning_type (str): Intended use of model, affects loss function and final
			activation function used. allowed: 'regression', 'binary_class',
			'multi_class'
		output_size (int): size of prediciton layer
		optimizer (str/keras.optimizer): used to compile the model
		kwargs: Used to call `build_graph_conv_net`

	# Returns:
		keras Model: compiled for learning_type

	'''

	# Define the input layers
	atoms = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
	bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
	edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')

	# Get output of main net
	net_output = build_graph_conv_net([atoms, bonds, edges], **kwargs)

	# Add final prediction layer
	learning_type = learning_type.lower()
	if learning_type == 'regression':
		final_activation = 'linear'
		loss='mse'
	elif learning_type == 'binary_class':
		final_activation = 'sigmoid'
		loss='binary_crossentropy'
	elif learning_type == 'multi_class':
		final_activation = 'softmax'
		loss='categorical_crossentropy'
	else:
		raise Exception('Invalid argument for learning type ({})'.format(learning_type))
	main_prediction = Dense(output_size, activation=final_activation, name='main_prediction')(net_output)

	# Build and compile the model
	model = models.Model(inputs=[atoms, bonds, edges], outputs=[main_prediction])
	model.compile(optimizer=optimizer, loss=loss)

	return model

def build_graph_conv_net(data_input, conv_layer_sizes=[], fp_layer_size=1024, net_layer_sizes=[], conv_activation='relu', fp_activation='softmax', net_activation='relu', conv_bias=True, fp_bias=True, net_bias=True, conv_l1=0, fp_l1=0, net_l1=0, conv_l2=0, fp_l2=0, net_l2=0, conv_dropout=0, fp_dropout=0, net_dropout=0, conv_batchnorm=0, fp_batchnorm=0, net_batchnorm=0, conv_kwargs={}, fp_kwargs={}, net_kwargs={}, fp_merge_mode='sum', atomwise_dropout=True, graphpool=False):
	''' Builds a graph convolutional network with a regular neural network on
		top.

	# Arguments
		data_input (tuple): The Input feature layers (as `[atoms, bonds, edges]`)

		# Layer sizes
			conv_layer_sizes (list): list of int sizes for each hidden layer in the graph
			fp_layer_size (int/list): list of int sizes for each output layer in the
				graph network. Should be either of length 1, or len(conv_layer_sizes)+1.
					When of lenght 1 (or as plain int), only one output layer will be
					added on top of the convolutional network.
					Otherwise, the first output layer will be connected to the unconvolved
					atom inputs, and the subsequent layers will be aligned with the
					hidden graph layers. In this case, use `None` for hidden layers
					where no fingerprint output is desired
			net_layer_sizes (list): list of int sizes for each hidden layer in the
				regular network on top of the graph network

		# Layer options
			All layer options are analougusly specified for the `NeuralGraphHidden`
			layers (prefix `conv_`), the `NeuralGraphOutput` layers (previx `fp_`)
			and the `Dense` layers of the regular net (prefix `net_`) respectively.

			All arguments can be specified as a single value or as a list of values,
			If a single value is specified, it will be used for all respective layers.
			If multiple values are specified, the list should have the same length
			as the layer_sizes list for the corresponding layers.

			activation (string/function): name of activation function to use,
				or alternatively, elementwise Theano function (see `keras.layers.Dense`)
			bias (bool): wether or not to use bias
			l1 (float): amount of l1 regularisation (use 0 for no l1 regularisation)
			l2 (float): amount of l2 regularisation (use 0 for no l2 regularisation)
			dropout (float): p value for dropout  (use 0 for no dropout)
			batchnorm (float): epsilon value for batchnorm (use 0 for no batchnorm)
			kwargs (dict): other arguments that will be passed to the `Dense` layers
				of the regular network, or to the inner `Dense` layers of the
				`NeuralGraph` layers

		fp_merge_mode (str): If multiple fingerprint ouput layers are specified,
			this arguments specifies how they are combined. (see `keras.layers.merge`)
			Note that if `fp_merge_mode='sum', all `fp_layer_size` should be equal.
		atomwise_dropout (bool): If true, the same atoms will be dropped out in
			each batch, this should be done because the weights are also shared
			between atoms in each batch. But could be turned of to investigate its
			effect
		graphpool (bool): If True, apply graphpool after each hidden graphlayer,
			can also be specified as a list

	# Returns:
		output (keras tensor): Ouput of final layer of network. Add a prediciton
			layer and use functional API to turn into a model
	'''

	# ======= Process network parameters =======
	# Rename for consistency
	fp_layer_sizes = fp_layer_size

	# Ensure fp_layer_sizes is a list
	if not is_iterable(fp_layer_sizes):
		fp_layer_sizes = [fp_layer_sizes]

	# Merge all parameter into tuples for each layer
	conv_layers = zip_mixed(conv_layer_sizes, conv_activation, conv_bias,
							conv_l1, conv_l2, conv_dropout, conv_batchnorm,
							conv_kwargs, graphpool, repeat_classes=[dict, str])
	fp_layers = zip_mixed(fp_layer_sizes, fp_activation, fp_bias,
							fp_l1, fp_l2, fp_dropout, fp_batchnorm,
							fp_kwargs, repeat_classes=[dict, str])
	net_layers = zip_mixed(net_layer_sizes, net_activation, net_bias,
							net_l1, net_l2, net_dropout, net_batchnorm,
							net_kwargs, repeat_classes=[dict, str])

	# Ensure fp_layers is of length conv_layers+1
	if len(fp_layer_sizes) != len(conv_layer_sizes)+1:
		assert len(fp_layer_sizes) == 1, 'Incorrect amount of fingerprint layers specified. Either specify 1 or len(conv_layer_sizes)+1 ({}) fp_layer_sizes ({})'.format(len(conv_layer_sizes)+1, len(fp_layer_sizes))
		# Add None for fp_layer_sizes and add None-tuples to fp_layers to align
		# 	fp layers with conv_layers (the one layer provided will be the last layer)
		fp_layer_sizes = [None]*len(conv_layer_sizes) + list(fp_layer_sizes)
		fp_layers = [(None, )*len(fp_layers[0])] *len(conv_layer_sizes) + fp_layers


	# Check zip result is the same length as specified sizes
	assert len(conv_layers) == len(conv_layer_sizes), 'If `conv_`-layer-arguments are specified as a list, they should have the same length as `conv_layer_sizes` (length {0}), found an argument of lenght {1}'.format(len(conv_layer_sizes), len(conv_layers))
	assert len(fp_layers) == len(fp_layer_sizes), 'If `fp`-layer-arguments are specified as a list, they should have the same length as `fp_layer_sizes` (len {0}), found an argument of lenght {1}'.format(len(fp_layer_sizes), len(fp_layers))
	assert len(net_layers) == len(net_layer_sizes), 'If `net_`-layer-arguments are specified as a list, they should have the same length as `net_layer_sizes` (length {0}), found an argument of lenght {1}'.format(len(net_layer_sizes), len(net_layers))

	# ======= Build the network =======

	# The inputs and outputs
	atoms, bonds, edges = data_input
	fingerprint_outputs = []

	def ConvDropout(p_dropout):
		''' Defines the standard Dropout layer for convnets
		'''
		if atomwise_dropout:
			return AtomwiseDropout(p_dropout)
		return Dropout(p_dropout)

	# Add first output layer directly to atom inputs
	fp_size, fp_activation, fp_bias, fp_l1, fp_l2, fp_dropout, fp_batchnorm, fp_kwargs = fp_layers.pop(0)
	if fp_size:
		fp_atoms_in = atoms

		if fp_batchnorm:
			fp_atoms_in = BatchNormalization(fp_batchnorm)(fp_atoms_in)
		if fp_dropout:
			fp_atoms_in = ConvDropout(fp_dropout)(fp_atoms_in)

		fp_out = NeuralGraphOutput(fp_size, activation=fp_activation, bias=fp_bias,
								   W_regularizer=l1_l2(fp_l1, fp_l2),
								   b_regularizer=l1_l2(fp_l1, fp_l2),
								   **fp_kwargs)([fp_atoms_in, bonds, edges])
		fingerprint_outputs.append(fp_out)

	# Add Graph convolutional layers
	convolved_atoms = [atoms]
	for conv_layer, fp_layer in zip(conv_layers, fp_layers):

		# Import parameters
		(conv_size, conv_activation, conv_bias, conv_l1, conv_l2, conv_dropout,
		 conv_batchnorm, conv_kwargs, graphpool) = conv_layer
		fp_size, fp_activation, fp_bias, fp_l1, fp_l2, fp_dropout, fp_batchnorm, fp_kwargs = fp_layer

		# Add hidden layer
		atoms_in = convolved_atoms[-1]

		if conv_batchnorm:
			atoms_in = BatchNormalization(conv_batchnorm)(atoms_in)
		if conv_dropout:
			atoms_in = ConvDropout(conv_dropout)(atoms_in)

		# Use inner_layer_fn init method of `NeuralGraphHidden`, because it is
		# 	the most powerfull (e.g. allows custom activation functions)
		def inner_layer_fn():
			return Dense(conv_size, activation=conv_activation, bias=conv_bias,
						 W_regularizer=l1_l2(conv_l1, conv_l2),
						 b_regularizer=l1_l2(conv_l1, conv_l2), **conv_kwargs)
		atoms_out = NeuralGraphHidden(inner_layer_fn)([atoms_in, bonds, edges])

		if graphpool:
			atoms_out = NeuralGraphPool()([atoms_out, bonds, edges])

		# Export
		convolved_atoms.append(atoms_out)

		# Add output layer (if specified)
		if fp_size:
			fp_atoms_in = atoms_out

			if fp_batchnorm:
				fp_atoms_in = BatchNormalization(fp_batchnorm)(fp_atoms_in)
			if fp_dropout:
				fp_atoms_in = ConvDropout(fp_dropout)(fp_atoms_in)

			fp_out = NeuralGraphOutput(fp_size, activation=fp_activation, bias=fp_bias,
									   W_regularizer=l1_l2(fp_l1, fp_l2),
									   b_regularizer=l1_l2(fp_l1, fp_l2),
									   **fp_kwargs)([fp_atoms_in, bonds, edges])

			# Export
			fingerprint_outputs.append(fp_out)



	# Merge fingerprint
	if len(fingerprint_outputs) > 1:
		final_fp = merge(fingerprint_outputs, mode=fp_merge_mode)
	else:
		final_fp = fingerprint_outputs[-1]

	# Add regular Neural net
	net_outputs = [final_fp]
	for net_layer in net_layers:

		# Import parameters
		layer_size, net_activation, net_bias, net_l1, net_l2, net_dropout, net_batchnorm, net_kwargs = net_layer

		# Add regular nn layers
		net_in = net_outputs[-1]

		if net_batchnorm:
			net_in = BatchNormalization(net_batchnorm)(net_in)
		if net_dropout:
			net_in = Dropout(net_dropout)(net_in)

		net_out = Dense(layer_size, activation=net_activation, bias=net_bias,
						W_regularizer=l1_l2(net_l1, net_l2),
						b_regularizer=l1_l2(net_l1, net_l2), **net_kwargs)(net_in)

		# Export
		net_outputs.append(net_out)

	return net_outputs[-1]