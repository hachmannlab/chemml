"""
Unified interface for neural graph convolutional fingerprints.

Supports both TensorFlow and PyTorch backends with automatic engine selection.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

from chemml.models.graphconvlayers import NeuralGraphHidden as NGH_TF
from chemml.models.graphconvlayers import NeuralGraphOutput as NGO_TF
from chemml.models.graphconvlayers_torch import _NeuralGraphNetworkPT


class NeuralGraphFingerprint:
    """Neural graph fingerprint model with engine selection.

    Generates fixed-size molecular fingerprints using graph neural networks.
    Supports lazy model building and automatic multi-output detection.

    Parameters
    ----------
    engine : str
        'tensorflow' or 'pytorch'
    conv_width : int, optional
        Width of convolution layers. Default 8.
    fp_length : int, optional
        Fingerprint output dimension. Default 200.
    n_conv_layers : int, optional
        Number of convolutional layers. Default 2.
    hidden_sizes : list, optional
        Sizes of optional MLP head layers. Default None (no MLP head).
    hidden_activations : list, optional
        Activations for MLP head. Default None.
    conv_activation : str, optional
        Activation for conv layers. Default 'relu'.
    fp_activation : str, optional
        Activation for fingerprint layers. Default 'softmax'.
    epochs : int, optional
        Training epochs. Default 100.
    batch_size : int, optional
        Batch size. Default 32.
    learning_rate : float, optional
        Learning rate. Default 0.001.
    optimizer : str, optional
        'adam' or 'sgd'. Default 'adam'.
    alpha : float, optional
        L2 regularization weight (PyTorch only). Default 0.001.
    use_bias : bool, optional
        Whether to use bias in layers. Default True.
    verbose : bool, optional
        Print training progress. Default True.
    random_seed : int, optional
        Random seed. Default None.
    """

    def __init__(
        self,
        engine,
        conv_width=8,
        fp_length=200,
        n_conv_layers=2,
        hidden_sizes=None,
        hidden_activations=None,
        conv_activation="relu",
        fp_activation="softmax",
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        alpha=0.001,
        use_bias=True,
        verbose=True,
        random_seed=None,
    ):
        if engine not in ["tensorflow", "pytorch"]:
            raise ValueError("engine must be 'tensorflow' or 'pytorch'")

        self.engine = engine
        self.conv_width = conv_width
        self.fp_length = fp_length
        self.n_conv_layers = n_conv_layers
        self.hidden_sizes = hidden_sizes
        self.hidden_activations = hidden_activations
        self.conv_activation = conv_activation
        self.fp_activation = fp_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.alpha = alpha
        self.use_bias = use_bias
        self.verbose = verbose
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            tf.random.set_seed(random_seed)

        self.model = None
        self.n_outputs = 1

    def _tensorize_input(self, molecules_input, max_degree=5, max_atoms=None, n_jobs=1, batch_size=10, verbose=False):
        """Convert SMILES or Molecule objects to tensor format.

        Uses normalize_input from foss_descriptors to ensure all inputs are in Molecule format,
        then tensorizes them for model input.

        Parameters
        ----------
        molecules_input : str, list, or Molecule
            Input molecules as:
            - SMILES string (single molecule)
            - List of SMILES strings
            - Molecule object (single molecule)
            - List of Molecule objects
        max_degree : int, optional
            Maximum degree for tensorization. Default 5.
        max_atoms : int, optional
            Maximum number of atoms. Default None (auto-detect from data).
        n_jobs : int, optional
            Number of parallel jobs for tensorization. Default 1.
        batch_size : int, optional
            Batch size for tensorization. Default 10.
        verbose : bool, optional
            Print progress during tensorization. Default False.

        Returns
        -------
        tuple
            (atoms, bonds, edges) as float32/int32 tensors
        """
        from chemml.chem.foss_descriptors import normalize_input
        from chemml.chem import tensorise_molecules

        # Normalize to Molecule format (handles SMILES, Molecule objects, etc.)
        # normalize_input with force_molecule=True returns (smiles_list, rdkit_mol_list, molecule_obj_list)
        _, _, mol_obj_list = normalize_input(
            molecules_input, quiet=not verbose, force_molecule=True
        )
        

        # Tensorize molecules
        atoms, bonds, edges = tensorise_molecules(
            molecules=mol_obj_list,
            max_degree=max_degree,
            max_atoms=max_atoms,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

        return atoms, bonds, edges

    def fit(self, atoms, bonds=None, edges=None, y=None, tensorize_kwargs=None, **kwargs):
        """Train the model.

        Supports both raw molecular inputs and pre-tensorized inputs.

        Parameters
        ----------
        atoms : array_like or str/list/Molecule
            Either:
            - Pre-tensorized atom features, shape (n_samples, max_atoms, num_atom_features)
            - Raw molecular input: SMILES string, list of SMILES, Molecule, or list of Molecules
        bonds : array_like, optional
            Bond features if atoms is pre-tensorized.
            Not needed if atoms is raw molecular input.
        edges : array_like, optional
            Edge connectivity if atoms is pre-tensorized.
            Not needed if atoms is raw molecular input.
        y : array_like
            Target values, shape (n_samples,) for single output or (n_samples, n_outputs) for multi-output
        tensorize_kwargs : dict, optional
            Arguments for _tensorize_input when using raw molecular input.
            Supported keys: max_degree, max_atoms, n_jobs, batch_size, verbose.
            Default: {max_degree=5, max_atoms=None, n_jobs=1, batch_size=10, verbose=False}

        Returns
        -------
        self

        Examples
        --------
        # Pre-tensorized input (existing workflow)
        ngf.fit(atoms, bonds, edges, y)

        # Raw SMILES input (new workflow)
        ngf.fit(['CC', 'CCO', 'CCCO'], y=y)

        # Raw Molecule objects (new workflow)
        ngf.fit(mol_objs_list, y=y)

        # With tensorization options
        ngf.fit(smiles_list, y=y, tensorize_kwargs={'max_degree': 5, 'n_jobs': -1})
        """
        # Detect input type and tensorize if needed
        
        if isinstance(atoms, list):
            # Raw molecular input - tensorize it
            if tensorize_kwargs is None:
                tensorize_kwargs = {}
                # Set defaults for tensorize_kwargs
                tensorize_kwargs.setdefault('max_degree', 5)
                tensorize_kwargs.setdefault('max_atoms', None)
                tensorize_kwargs.setdefault('n_jobs', 1)
                tensorize_kwargs.setdefault('batch_size', 10)
                tensorize_kwargs.setdefault('verbose', self.verbose)
                # print(f"[Info] Using default tensorize_kwargs: {tensorize_kwargs}")

            atoms, bonds, edges = self._tensorize_input(atoms, **tensorize_kwargs)
            # print(atoms.shape, bonds.shape, edges.shape)
        elif isinstance(atoms, np.ndarray):
            if bonds is None or edges is None:
                raise ValueError(
                    "If atoms is a pre-tensorized array, bonds and edges must also be provided. "
                    "Alternatively, pass raw molecular input (SMILES/Molecule objects) as atoms."
                )
        else:
            raise ValueError(
                "atoms must be either a list of SMILES/Molecule objects or a pre-tensorized numpy array alongside edges and bonds."
            )

        # Extract tensor shapes
        max_atoms = atoms.shape[1]
        max_degree = edges.shape[2]
        num_atom_features = atoms.shape[2]
        num_bond_features = bonds.shape[3]

        # Detect number of outputs from y
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            detected_n_outputs = 1
        else:
            detected_n_outputs = y_array.shape[1]

        # Build model if not already built or if output size changed
        if self.model is None:
            self._build_model(
                max_atoms, max_degree, num_atom_features, num_bond_features, detected_n_outputs
            )
        elif detected_n_outputs != self.n_outputs:
            # Rebuild output layer for new output count
            self._rebuild_output_layer(detected_n_outputs)

        # Dispatch to appropriate training method
        if self.engine == "tensorflow":
            self._fit_tensorflow(atoms, bonds, edges, y_array)
        else:
            self._fit_pytorch(atoms, bonds, edges, y_array)

        return self

    def _build_model(self, max_atoms, max_degree, num_atom_features, num_bond_features, n_outputs):
        """Build the model architecture.

        Parameters
        ----------
        max_atoms : int
        max_degree : int
        num_atom_features : int
        num_bond_features : int
        n_outputs : int
        """
        self.n_outputs = n_outputs

        if self.engine == "tensorflow":
            self._build_tensorflow(
                max_atoms, max_degree, num_atom_features, num_bond_features, n_outputs
            )
        else:
            self._build_pytorch(
                max_degree, num_atom_features, num_bond_features, n_outputs
            )

    def _build_tensorflow(self, max_atoms, max_degree, num_atom_features, num_bond_features, n_outputs):
        """Build TensorFlow model."""
        # Input layers
        atoms_input = Input(
            shape=(max_atoms, num_atom_features), name="atom_inputs", batch_size=None
        )
        bonds_input = Input(
            shape=(max_atoms, max_degree, num_bond_features), name="bond_inputs", batch_size=None
        )
        edges_input = Input(
            shape=(max_atoms, max_degree), name="edge_inputs", dtype="int32", batch_size=None
        )

        # Build convolution layers and collect fingerprints
        fingerprints = []

        # Fingerprint from raw atoms
        fp0 = NGO_TF(
            self.fp_length, activation=self.fp_activation, use_bias=self.use_bias
        )([atoms_input, bonds_input, edges_input])
        fingerprints.append(fp0)

        # Convolutional layers
        current_atoms = atoms_input
        for i in range(self.n_conv_layers):
            current_atoms = NGH_TF(
                self.conv_width,
                activation=self.conv_activation,
                use_bias=self.use_bias,
            )([current_atoms, bonds_input, edges_input])

            fp_i = NGO_TF(
                self.fp_length, activation=self.fp_activation, use_bias=self.use_bias
            )([current_atoms, bonds_input, edges_input])
            fingerprints.append(fp_i)

        # Sum fingerprints
        if len(fingerprints) > 1:
            combined_fp = Add()(fingerprints)
        else:
            combined_fp = fingerprints[0]

        # Optional MLP head
        if self.hidden_sizes is not None:
            x = combined_fp
            for size, activation in zip(self.hidden_sizes, self.hidden_activations or []):
                x = Dense(
                    size,
                    activation=activation.lower() if activation else None,
                    kernel_regularizer=None if self.engine == "pytorch" else None,
                )(x)
            output = Dense(n_outputs, activation="linear")(x)
        else:
            output = Dense(n_outputs, activation="linear")(combined_fp)

        # Build model
        self.model = Model(
            inputs=[atoms_input, bonds_input, edges_input], outputs=output
        )

        # Compile
        if self.optimizer == "adam":
            opt = Adam(learning_rate=self.learning_rate)
        else:
            opt = SGD(learning_rate=self.learning_rate)

        self.model.compile(optimizer=opt, loss="mse")

    def _build_pytorch(self, max_degree, num_atom_features, num_bond_features, n_outputs):
        """Build PyTorch model."""
        self.model = _NeuralGraphNetworkPT(
            conv_width=self.conv_width,
            fp_length=self.fp_length,
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            max_degree=max_degree,
            n_conv_layers=self.n_conv_layers,
            hidden_sizes=self.hidden_sizes,
            hidden_activations=self.hidden_activations,
            n_outputs=n_outputs,
            conv_activation=self.conv_activation,
            fp_activation=self.fp_activation,
            use_bias=self.use_bias,
        )

        # Create optimizer
        if self.optimizer == "adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.alpha,
            )
        elif self.optimizer == "sgd":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.alpha,
            )
        else:
            raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

        self.losses_ = []

    def _rebuild_output_layer(self, n_outputs):
        """Rebuild output layer for multi-output support.

        Parameters
        ----------
        n_outputs : int
            New number of outputs
        """
        if self.engine == "pytorch":
            self.model._rebuild_output_layer(n_outputs)
        else:
            # TensorFlow: rebuild entire model
            # Extract shapes from current model
            atoms_shape = self.model.input_shape[0]
            bonds_shape = self.model.input_shape[1]
            edges_shape = self.model.input_shape[2]

            max_atoms = atoms_shape[1]
            max_degree = edges_shape[2]
            num_atom_features = atoms_shape[2]
            num_bond_features = bonds_shape[3]

            self._build_tensorflow(
                max_atoms, max_degree, num_atom_features, num_bond_features, n_outputs
            )

    def _fit_tensorflow(self, atoms, bonds, edges, y):
        """Train TensorFlow model."""
        self.model.fit(
            [atoms, bonds, edges],
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1 if self.verbose else 0,
        )

    def _fit_pytorch(self, atoms, bonds, edges, y):
        """Train PyTorch model."""
        # Convert to tensors
        atoms_t = torch.tensor(atoms, dtype=torch.float32)
        bonds_t = torch.tensor(bonds, dtype=torch.float32)
        edges_t = torch.tensor(edges, dtype=torch.int32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Ensure y has correct shape
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        n_samples = atoms_t.shape[0]
        self.losses_ = []

        # Training loop
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training", disable=not self.verbose):
            # Shuffle
            permutation = torch.randperm(n_samples)
            epoch_losses = []

            # Mini-batches
            for i in range(0, n_samples, self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_atoms = atoms_t[indices]
                batch_bonds = bonds_t[indices]
                batch_edges = edges_t[indices]
                batch_y = y_t[indices]

                # Forward pass
                self.opt.zero_grad()
                y_pred = self.model(batch_atoms, batch_bonds, batch_edges)

                # Ensure shapes match
                if y_pred.ndim == 1:
                    y_pred = y_pred.unsqueeze(1)

                # Loss
                loss = torch.nn.functional.mse_loss(y_pred, batch_y)

                # Backward
                loss.backward()
                self.opt.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.losses_.append(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

    def predict(self, atoms, bonds=None, edges=None, tensorize_kwargs=None):
        """Generate predictions.

        Supports both raw molecular inputs and pre-tensorized inputs.

        Parameters
        ----------
        atoms : array_like or str/list/Molecule
            Either:
            - Pre-tensorized atom features, shape (n_samples, max_atoms, num_atom_features)
            - Raw molecular input: SMILES string, list of SMILES, Molecule, or list of Molecules
        bonds : array_like, optional
            Bond features if atoms is pre-tensorized.
            Not needed if atoms is raw molecular input.
        edges : array_like, optional
            Edge connectivity if atoms is pre-tensorized.
            Not needed if atoms is raw molecular input.
        tensorize_kwargs : dict, optional
            Arguments for _tensorize_input when using raw molecular input.
            Supported keys: max_degree, max_atoms, n_jobs, batch_size, verbose.

        Returns
        -------
        predictions : ndarray
            Predictions, shape (n_samples,) for single output or (n_samples, n_outputs) for multi-output

        Examples
        --------
        # Pre-tensorized input
        predictions = ngf.predict(atoms_test, bonds_test, edges_test)

        # Raw SMILES input
        predictions = ngf.predict(['CC', 'CCO', 'CCCO'])

        # Raw Molecule objects
        predictions = ngf.predict(mol_objs_test)
        """
        if self.model is None:
            raise ValueError("Model not built. Call fit() first.")

        # Detect input type and tensorize if needed
        if isinstance(atoms, (str, list)) or (hasattr(atoms, '__class__') and 
                                               atoms.__class__.__name__ == 'Molecule'):
            # Raw molecular input - tensorize it
            if tensorize_kwargs is None:
                tensorize_kwargs = {}
            # Set defaults for tensorize_kwargs
            tensorize_kwargs.setdefault('max_degree', 5)
            tensorize_kwargs.setdefault('max_atoms', None)
            tensorize_kwargs.setdefault('n_jobs', 1)
            tensorize_kwargs.setdefault('batch_size', 10)
            tensorize_kwargs.setdefault('verbose', False)

            atoms, bonds, edges = self._tensorize_input(atoms, **tensorize_kwargs)
        elif bonds is None or edges is None:
            raise ValueError(
                "If atoms is a pre-tensorized array, bonds and edges must also be provided. "
                "Alternatively, pass raw molecular input (SMILES/Molecule objects) as atoms."
            )

        if self.engine == "tensorflow":
            predictions = self.model.predict([atoms, bonds, edges], verbose=0)
        else:
            # PyTorch
            atoms_t = torch.tensor(atoms, dtype=torch.float32)
            bonds_t = torch.tensor(bonds, dtype=torch.float32)
            edges_t = torch.tensor(edges, dtype=torch.int32)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(atoms_t, bonds_t, edges_t)
                predictions = predictions.numpy()

        # Ensure correct output shape
        if self.n_outputs == 1 and predictions.ndim > 1:
            predictions = predictions.squeeze(axis=1)

        return predictions

    def get_model(self):
        """Return the underlying model object.

        Returns
        -------
        model : tensorflow.keras.Model or _NeuralGraphNetworkPT
            The underlying model
        """
        return self.model
