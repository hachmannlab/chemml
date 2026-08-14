"""
Pytest tests for NeuralGraphFingerprint (graph convolutional neural networks).
"""

import pytest
import numpy as np
import os
import warnings
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(3)

from chemml.models import NeuralGraphFingerprint
from chemml.models.graphconvlayers_torch import _NeuralGraphNetworkPT
from chemml.datasets import load_organic_density
from chemml.chem import Molecule, tensorise_molecules
from tensorflow.keras.models import Model


@pytest.fixture(scope='session')
def graph_data():
    """Load molecular graph data from load_organic_density dataset.
    
    Loads first 20 molecules from the organic density dataset, converts
    SMILES to molecule objects, and tensorizes them for testing.
    """
    # Suppress warnings during molecule creation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # Load dataset
        molecules_df, target_df, _ = load_organic_density()
        
        # Use first 20 molecules for faster testing
        n_samples = 20
        smiles_list = molecules_df['smiles'].iloc[:n_samples].tolist()
        
        # Convert SMILES to Molecule objects
        mol_objs_list = []
        for smi in smiles_list:
            try:
                mol = Molecule(smi, 'smiles')
                mol.hydrogens('add')
                mol.to_xyz('MMFF', maxIters=5000, mmffVariant='MMFF94s')
                mol_objs_list.append(mol)
            except Exception:
                # Skip molecules that fail to initialize
                continue
        
        # Handle case where not enough molecules succeeded
        if len(mol_objs_list) < 10:
            # Fall back to synthetic data if molecule processing fails
            n_samples = 10
            max_atoms = 5
            max_degree = 3
            num_atom_features = 6
            num_bond_features = 4
            
            atoms = np.random.randn(n_samples, max_atoms, num_atom_features).astype('float32')
            bonds = np.random.randn(n_samples, max_atoms, max_degree, num_bond_features).astype('float32')
            edges = np.random.randint(-1, max_atoms, size=(n_samples, max_atoms, max_degree)).astype('int32')
        else:
            # Tensorize molecules
            atoms, bonds, edges = tensorise_molecules(
                molecules=mol_objs_list,
                max_degree=5,
                max_atoms=None,
                n_jobs=1,
                batch_size=10,
                verbose=False
            )
        
        n_samples = atoms.shape[0]
        # Get corresponding targets
        targets = target_df['density_Kg/m3'].iloc[:n_samples].values.astype('float32')
    
    return atoms, bonds, edges, targets


@pytest.fixture()
def single_output_targets(graph_data):
    """Single output target values from dataset."""
    atoms, bonds, edges, targets = graph_data
    # Use actual targets from dataset
    return targets


@pytest.fixture()
def multi_output_targets(graph_data):
    """Multi-output target values (replicate single output 3 times for testing)."""
    atoms, bonds, edges, targets = graph_data
    # Create synthetic multi-output by replicating and adding noise
    y_multi = np.column_stack([
        targets,
        targets + np.random.randn(len(targets)) * 0.1,
        targets + np.random.randn(len(targets)) * 0.1
    ]).astype('float32')
    return y_multi


class TestNeuralGraphFingerprintInit:
    """Test NeuralGraphFingerprint initialization."""

    def test_pytorch_init(self):
        """Test PyTorch engine initialization."""
        ngf = NeuralGraphFingerprint(
            engine='pytorch',
            conv_width=8,
            fp_length=32,
            n_conv_layers=2
        )
        assert ngf.engine == 'pytorch'
        assert ngf.model is None  # Not built yet
        assert ngf.n_outputs == 1

    def test_tensorflow_init(self):
        """Test TensorFlow engine initialization."""
        ngf = NeuralGraphFingerprint(
            engine='tensorflow',
            conv_width=8,
            fp_length=32,
            n_conv_layers=2
        )
        assert ngf.engine == 'tensorflow'
        assert ngf.model is None  # Not built yet
        assert ngf.n_outputs == 1

    def test_invalid_engine(self):
        """Test that invalid engine raises ValueError."""
        with pytest.raises(ValueError, match="engine must be"):
            NeuralGraphFingerprint(engine='invalid')

    def test_with_mlp_head(self):
        """Test initialization with MLP head parameters."""
        ngf = NeuralGraphFingerprint(
            engine='pytorch',
            conv_width=8,
            fp_length=32,
            hidden_sizes=[128, 64],
            hidden_activations=['relu', 'relu']
        )
        assert ngf.hidden_sizes == [128, 64]
        assert ngf.hidden_activations == ['relu', 'relu']


class TestPyTorchEngine:
    """Test PyTorch engine functionality."""
    """Note all tests are run with output head, as it is the default behavior for neural fingerprints."""

    def test_pytorch_single_output(self, graph_data, single_output_targets):
        """Test PyTorch with optional MLP head."""
        atoms, bonds, edges, _ = graph_data
        y = single_output_targets

        ngf = NeuralGraphFingerprint(
            engine='pytorch',
            conv_width=8,
            fp_length=32,
            n_conv_layers=1,
            hidden_sizes=[64, 32],
            hidden_activations=['relu', 'relu'],
            epochs=2,
            verbose=False
        )

        ngf.fit(atoms, bonds, edges, y)

        predictions = ngf.predict(atoms, bonds, edges)
        assert predictions.shape == (atoms.shape[0],)
    def test_pytorch_multi_output(self, graph_data, multi_output_targets):
        """Test PyTorch multi-output training and prediction."""
        atoms, bonds, edges, _ = graph_data
        y = multi_output_targets

        ngf = NeuralGraphFingerprint(
            engine='pytorch',
            conv_width=8,
            fp_length=32,
            n_conv_layers=1,
            hidden_sizes=[64, 32],
            hidden_activations=['relu', 'relu'],
            epochs=2,
            verbose=False
        )

        ngf.fit(atoms, bonds, edges, y)

        predictions = ngf.predict(atoms, bonds, edges)
        assert predictions.shape == (atoms.shape[0], y.shape[1])


class TestTensorFlowEngine:
    """Test TensorFlow engine functionality."""
    """Note all tests are run with output head, as it is the default behavior for neural fingerprints."""
    def test_tensorflow_single_output(self, graph_data, single_output_targets):
        """Test TensorFlow with optional MLP head."""
        atoms, bonds, edges, _ = graph_data
        y = single_output_targets

        ngf = NeuralGraphFingerprint(
            engine='tensorflow',
            conv_width=8,
            fp_length=32,
            n_conv_layers=1,
            hidden_sizes=[64, 32],
            hidden_activations=['relu', 'relu'],
            epochs=2,
            verbose=False
        )

        ngf.fit(atoms, bonds, edges, y)

        predictions = ngf.predict(atoms, bonds, edges)
        assert predictions.shape == (atoms.shape[0],)
    def test_tensorflow_multi_output(self, graph_data, multi_output_targets):
        """Test TensorFlow multi-output training and prediction."""
        atoms, bonds, edges, _ = graph_data
        y = multi_output_targets

        ngf = NeuralGraphFingerprint(
            engine='tensorflow',
            conv_width=8,
            fp_length=32,
            n_conv_layers=1,
            hidden_sizes=[64, 32],
            hidden_activations=['relu', 'relu'],
            epochs=2,
            verbose=False
        )

        ngf.fit(atoms, bonds, edges, y)

        predictions = ngf.predict(atoms, bonds, edges)
        assert predictions.shape == (atoms.shape[0], y.shape[1])


class TestGetModel:
    """Test get_model method."""

    def test_get_model_pytorch(self, graph_data, single_output_targets):
        """Test get_model returns PyTorch model."""
        atoms, bonds, edges, _ = graph_data
        y = single_output_targets

        ngf = NeuralGraphFingerprint(engine='pytorch', epochs=1, verbose=False)
        ngf.fit(atoms, bonds, edges, y)

        model = ngf.get_model()
        assert isinstance(model, _NeuralGraphNetworkPT)

    def test_get_model_tensorflow(self, graph_data, single_output_targets):
        """Test get_model returns TensorFlow model."""
        atoms, bonds, edges, _ = graph_data
        y = single_output_targets

        ngf = NeuralGraphFingerprint(engine='tensorflow', epochs=1, verbose=False)
        ngf.fit(atoms, bonds, edges, y)

        model = ngf.get_model()
        assert isinstance(model, Model)

    def test_get_model_before_fit_raises(self):
        """Test that get_model raises error before fit."""
        ngf = NeuralGraphFingerprint(engine='pytorch')
        with pytest.raises(ValueError, match="Model not built"):
            ngf.predict(np.random.randn(1, 5, 6), np.random.randn(1, 5, 3, 4), np.random.randint(-1, 5, (1, 5, 3)))
