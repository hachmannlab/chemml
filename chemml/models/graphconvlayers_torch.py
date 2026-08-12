"""
PyTorch implementation of neural graph convolutional layers.

Adapted from TensorFlow/Keras graphconvlayers.py

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def neighbour_lookup(atoms, edges, maskvalue=0, include_self=False):
    """Look up atom features for neighbors in a batch of molecules.

    Parameters
    ----------
    atoms : torch.Tensor
        Shape (batch_n, max_atoms, num_atom_features)
    edges : torch.Tensor
        Shape (batch_n, max_atoms, max_degree) with neighbor indices and -1 as padding
    maskvalue : float, optional
        Value to use for empty/padded atoms. Default 0.
    include_self : bool, optional
        If True, include the atom's own features. Default False.

    Returns
    -------
    torch.Tensor
        Shape (batch_n, max_atoms(+1), max_degree, num_atom_features)
    """
    batch_n, max_atoms, max_degree = edges.shape
    num_atom_features = atoms.shape[-1]

    # Shift indices: -1 (padding) becomes 0, valid indices increment by 1
    masked_edges = edges + 1

    # Prepend a row of maskvalue at index 0 to handle invalid indices
    padded_atoms = torch.full(
        (batch_n, 1, num_atom_features),
        maskvalue,
        dtype=atoms.dtype,
        device=atoms.device
    )
    masked_atoms = torch.cat([padded_atoms, atoms], dim=1)  # (batch_n, max_atoms+1, num_atom_features)

    # Flatten for batch-aware gathering
    flattened_atoms = masked_atoms.reshape(-1, num_atom_features)  # ((batch_n*(max_atoms+1)), num_atom_features)
    lookup_size = masked_atoms.shape[1]

    # Create batch offset for flat indexing
    batch_offset = (torch.arange(batch_n, dtype=masked_edges.dtype, device=masked_edges.device) * lookup_size)
    batch_offset = batch_offset.reshape(batch_n, 1, 1)  # (batch_n, 1, 1)

    # Apply offset to flat edge indices
    flat_edges = (masked_edges + batch_offset).reshape(batch_n, -1)  # (batch_n, max_atoms*max_degree)

    # Gather neighbors
    gathered = flattened_atoms[flat_edges.long()]  # (batch_n*max_atoms*max_degree, num_atom_features)
    output = gathered.reshape(batch_n, max_atoms, max_degree, num_atom_features)

    if include_self:
        # Add atom's own features at the beginning
        atoms_expanded = atoms.unsqueeze(2)  # (batch_n, max_atoms, 1, num_atom_features)
        output = torch.cat([atoms_expanded, output], dim=2)  # (batch_n, max_atoms, max_degree+1, num_atom_features)

    return output


class NeuralGraphHidden(nn.Module):
    """Hidden convolutional layer for neural graphs.

    Applies degree-aware graph convolution. For each atom, features from neighbors
    are aggregated, concatenated with bond features, and passed through a separate
    dense layer for each degree.

    Parameters
    ----------
    conv_width : int
        Output feature dimension
    num_atom_features : int
        Input atom feature dimension
    num_bond_features : int
        Bond feature dimension
    max_degree : int
        Maximum node degree (number of neighbors)
    activation : str, optional
        Activation function name. Default 'relu'.
    use_bias : bool, optional
        Whether to use bias in linear layers. Default True.
    """

    def __init__(
        self,
        conv_width,
        num_atom_features,
        num_bond_features,
        max_degree,
        activation="relu",
        use_bias=True,
    ):
        super().__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.activation = activation.lower()
        self.input_dim = num_atom_features + num_bond_features

        # One linear layer per degree
        self.degree_layers = nn.ModuleList([
            nn.Linear(self.input_dim, conv_width, bias=use_bias)
            for _ in range(max_degree)
        ])

    def forward(self, atoms, bonds, edges):
        """Forward pass.

        Parameters
        ----------
        atoms : torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        bonds : torch.Tensor
            Shape (batch_n, max_atoms, max_degree, num_bond_features)
        edges : torch.Tensor
            Shape (batch_n, max_atoms, max_degree), dtype int32

        Returns
        -------
        torch.Tensor
            Shape (batch_n, max_atoms, conv_width)
        """
        batch_n, max_atoms, num_atom_features = atoms.shape
        num_bond_features = bonds.shape[-1]

        # Compute degree per atom (number of valid neighbors)
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True).float()  # (batch_n, max_atoms, 1)

        # Look up neighbor atom features
        neighbor_atom_features = neighbour_lookup(atoms, edges, include_self=True)  # (..., max_atoms, max_degree+1, num_atom_features)
        
        # Sum over neighbor dimension
        summed_atom_features = neighbor_atom_features.sum(dim=2)  # (batch_n, max_atoms, num_atom_features)

        # Sum bond features over degree dimension
        summed_bond_features = bonds.sum(dim=2)  # (batch_n, max_atoms, num_bond_features)

        # Concatenate atom and bond features
        combined = torch.cat([summed_atom_features, summed_bond_features], dim=-1)  # (batch_n, max_atoms, input_dim)

        # Process by degree: different linear layer per degree
        new_features_by_degree = []
        for degree in range(self.max_degree):
            # Mask for atoms of this degree
            degree_mask = (atom_degrees == degree).float()  # (batch_n, max_atoms, 1)

            # Apply linear transformation
            degree_out = self.degree_layers[degree](combined)  # (batch_n, max_atoms, conv_width)

            # Apply activation
            if self.activation == "relu":
                degree_out = F.relu(degree_out)
            elif self.activation == "sigmoid":
                degree_out = torch.sigmoid(degree_out)
            elif self.activation == "tanh":
                degree_out = torch.tanh(degree_out)
            elif self.activation == "softmax":
                degree_out = F.softmax(degree_out, dim=-1)

            # Apply mask
            degree_out = degree_out * degree_mask

            new_features_by_degree.append(degree_out)

        # Sum contributions from all degrees
        output = torch.stack(new_features_by_degree, dim=0).sum(dim=0)  # (batch_n, max_atoms, conv_width)

        return output


class NeuralGraphOutput(nn.Module):
    """Output layer for neural graph fingerprints.

    Converts atom features to a fixed-size fingerprint by applying a linear
    transformation to each atom and summing over all atoms (weighted by atom validity).

    Parameters
    ----------
    fp_length : int
        Fingerprint output dimension
    num_atom_features : int
        Input atom feature dimension
    num_bond_features : int
        Bond feature dimension
    activation : str, optional
        Activation function name. Default 'softmax'.
    use_bias : bool, optional
        Whether to use bias in linear layer. Default True.
    """

    def __init__(
        self,
        fp_length,
        num_atom_features,
        num_bond_features,
        activation="softmax",
        use_bias=True,
    ):
        super().__init__()
        self.fp_length = fp_length
        self.activation = activation.lower()
        self.input_dim = num_atom_features + num_bond_features
        self.linear = nn.Linear(self.input_dim, fp_length, bias=use_bias)

    def forward(self, atoms, bonds, edges):
        """Forward pass.

        Parameters
        ----------
        atoms : torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        bonds : torch.Tensor
            Shape (batch_n, max_atoms, max_degree, num_bond_features)
        edges : torch.Tensor
            Shape (batch_n, max_atoms, max_degree), dtype int32

        Returns
        -------
        torch.Tensor
            Shape (batch_n, fp_length)
        """
        batch_n, max_atoms, num_atom_features = atoms.shape

        # Atom mask: atoms that have at least one neighbor
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True).float()  # (batch_n, max_atoms, 1)
        atom_mask = (atom_degrees > 0).float()  # (batch_n, max_atoms, 1)

        # Sum bond features over degree
        summed_bond_features = bonds.sum(dim=2)  # (batch_n, max_atoms, num_bond_features)

        # Concatenate atom and bond features
        combined = torch.cat([atoms, summed_bond_features], dim=-1)  # (batch_n, max_atoms, input_dim)

        # Apply linear layer to each atom
        fp_unmasked = self.linear(combined)  # (batch_n, max_atoms, fp_length)

        # Apply activation
        if self.activation == "softmax":
            fp_unmasked = F.softmax(fp_unmasked, dim=-1)
        elif self.activation == "relu":
            fp_unmasked = F.relu(fp_unmasked)
        elif self.activation == "sigmoid":
            fp_unmasked = torch.sigmoid(fp_unmasked)
        elif self.activation == "tanh":
            fp_unmasked = torch.tanh(fp_unmasked)

        # Apply mask
        fp_masked = fp_unmasked * atom_mask  # (batch_n, max_atoms, fp_length)

        # Sum over atoms to get fingerprint
        fingerprint = fp_masked.sum(dim=1)  # (batch_n, fp_length)

        return fingerprint


class NeuralGraphPool(nn.Module):
    """Pooling layer for neural graphs.

    For each atom, computes the max of features across itself and all neighbors.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    def forward(self, atoms, bonds, edges):
        """Forward pass.

        Parameters
        ----------
        atoms : torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        bonds : torch.Tensor
            Shape (batch_n, max_atoms, max_degree, num_bond_features)
        edges : torch.Tensor
            Shape (batch_n, max_atoms, max_degree), dtype int32

        Returns
        -------
        torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        """
        # Look up neighbor features with -inf for padding
        neighbor_features = neighbour_lookup(atoms, edges, maskvalue=float('-inf'), include_self=True)

        # Max pool over neighbors
        max_features, _ = neighbor_features.max(dim=2)  # (batch_n, max_atoms, num_atom_features)

        # Apply atom mask (atoms with no neighbors get zeroed)
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True).float()
        atom_mask = (atom_degrees > 0).float()

        return max_features * atom_mask


class AtomwiseDropout(nn.Module):
    """Dropout applied uniformly across all atoms.

    Each atom shares the same dropout mask across its feature dimension.

    Parameters
    ----------
    p : float
        Dropout probability
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, atoms):
        """Forward pass.

        Parameters
        ----------
        atoms : torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)

        Returns
        -------
        torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        """
        if not self.training or self.p == 0:
            return atoms

        batch_n, max_atoms, num_features = atoms.shape

        # Create dropout mask for features only (shared across atoms)
        feature_mask = F.dropout(
            torch.ones(batch_n, num_features, device=atoms.device),
            p=self.p,
            training=True
        )  # (batch_n, num_features)

        # Expand and apply to all atoms
        feature_mask = feature_mask.unsqueeze(1)  # (batch_n, 1, num_features)
        return atoms * feature_mask


class _NeuralGraphNetworkPT(nn.Module):
    """Full PyTorch neural graph network model.

    Stacks multiple graph convolution layers and outputs fingerprints.

    Parameters
    ----------
    conv_width : int
        Width of convolution layers
    fp_length : int
        Fingerprint dimension
    num_atom_features : int
        Initial atom feature dimension
    num_bond_features : int
        Bond feature dimension
    max_degree : int
        Maximum node degree
    n_conv_layers : int, optional
        Number of convolution layers. Default 2.
    hidden_sizes : list, optional
        Sizes of MLP head layers. Default None (no MLP head).
    hidden_activations : list, optional
        Activation functions for MLP layers. Default None.
    n_outputs : int, optional
        Number of output dimensions. Default 1.
    conv_activation : str, optional
        Activation for convolution layers. Default 'relu'.
    fp_activation : str, optional
        Activation for fingerprint layers. Default 'softmax'.
    use_bias : bool, optional
        Whether to use bias. Default True.
    """

    def __init__(
        self,
        conv_width,
        fp_length,
        num_atom_features,
        num_bond_features,
        max_degree,
        n_conv_layers=2,
        hidden_sizes=None,
        hidden_activations=None,
        n_outputs=1,
        conv_activation="relu",
        fp_activation="softmax",
        use_bias=True,
    ):
        super().__init__()
        self.conv_width = conv_width
        self.fp_length = fp_length
        self.n_outputs = n_outputs
        self.max_degree = max_degree

        # Build convolution layers
        self.hidden_layers = nn.ModuleList()
        current_input_dim = num_atom_features

        for i in range(n_conv_layers):
            layer = NeuralGraphHidden(
                conv_width,
                current_input_dim,
                num_bond_features,
                max_degree,
                activation=conv_activation,
                use_bias=use_bias,
            )
            self.hidden_layers.append(layer)
            current_input_dim = conv_width

        # Build fingerprint output layers (one per conv level + raw input)
        self.fp_layers = nn.ModuleList()
        for i in range(n_conv_layers + 1):
            if i == 0:
                input_dim = num_atom_features
            else:
                input_dim = conv_width
            layer = NeuralGraphOutput(
                fp_length,
                input_dim,
                num_bond_features,
                activation=fp_activation,
                use_bias=use_bias,
            )
            self.fp_layers.append(layer)

        # Build MLP head if provided
        if hidden_sizes is not None:
            mlp_layers = []
            prev_size = fp_length
            for size, activation in zip(hidden_sizes, hidden_activations or []):
                mlp_layers.append(nn.Linear(prev_size, size, bias=use_bias))
                if activation:
                    if activation.lower() == "relu":
                        mlp_layers.append(nn.ReLU())
                    elif activation.lower() == "sigmoid":
                        mlp_layers.append(nn.Sigmoid())
                    elif activation.lower() == "tanh":
                        mlp_layers.append(nn.Tanh())
                prev_size = size
            self.mlp_head = nn.Sequential(*mlp_layers)
            final_input_dim = prev_size
        else:
            self.mlp_head = None
            final_input_dim = fp_length

        # Output layer
        self.output_layer = nn.Linear(final_input_dim, n_outputs, bias=use_bias)

    def forward(self, atoms, bonds, edges):
        """Forward pass.

        Parameters
        ----------
        atoms : torch.Tensor
            Shape (batch_n, max_atoms, num_atom_features)
        bonds : torch.Tensor
            Shape (batch_n, max_atoms, max_degree, num_bond_features)
        edges : torch.Tensor
            Shape (batch_n, max_atoms, max_degree), dtype int32

        Returns
        -------
        torch.Tensor
            Shape (batch_n, n_outputs) or (batch_n,) if n_outputs=1
        """
        # Compute fingerprints at each conv level
        fingerprints = []

        # Fingerprint from raw atoms
        fingerprints.append(self.fp_layers[0](atoms, bonds, edges))

        # Fingerprints from each hidden layer
        current_atoms = atoms
        for i, hidden_layer in enumerate(self.hidden_layers):
            current_atoms = hidden_layer(current_atoms, bonds, edges)
            fingerprints.append(self.fp_layers[i + 1](current_atoms, bonds, edges))

        # Sum all fingerprints
        combined_fp = torch.stack(fingerprints, dim=0).sum(dim=0)  # (batch_n, fp_length)

        # MLP head
        if self.mlp_head is not None:
            combined_fp = self.mlp_head(combined_fp)

        # Output layer
        output = self.output_layer(combined_fp)  # (batch_n, n_outputs)

        # Squeeze if single output
        if self.n_outputs == 1:
            output = output.squeeze(-1)

        return output

    def _rebuild_output_layer(self, n_outputs):
        """Rebuild output layer for multi-output support.

        Parameters
        ----------
        n_outputs : int
            New number of output dimensions
        """
        if self.mlp_head is not None:
            final_input_dim = list(self.mlp_head.modules())[-1].out_features
        else:
            final_input_dim = self.fp_length

        self.n_outputs = n_outputs
        self.output_layer = nn.Linear(final_input_dim, n_outputs, bias=True)
