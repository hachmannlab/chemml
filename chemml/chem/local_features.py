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
from __future__ import division, print_function

import numpy as np
from rdkit import Chem
from functools import partial
from multiprocessing import cpu_count, Pool

from keras.utils.generic_utils import Progbar


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)


def atom_features(atom):
    """
    This function encodes the RDKit atom to a binary vector.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        The bond must be an RDKit Bond object.

    Returns
    -------
    ndarray:
        A binary array with length 6 that specifies the type of bond, if it is
        a single/double/triple/aromatic bond, a conjugated bond or belongs to a molecular ring.

    """
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])


def bond_features(bond):
    """
    This function encodes the RDKit bond to a binary vector.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        The bond must be an RDKit Bond object.

    Returns
    -------
    ndarray:
        A binary array with length 6 that specifies the type of bond, if it is
        a single/double/triple/aromatic bond, a conjugated bond or belongs to a molecular ring.

    """
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


def num_atom_features():
    """
    This function returns the number of atomic features that are available by this module.

    Returns
    -------
    int:
        length of atomic feature vector.
    """
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    """
    This function returns the number of bond features that are available by this module.

    Returns
    -------
    int:
        length of bond feature vector.
    """
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    ''' Padds one axis of an array to a new size

    This is just a wrapper for np.pad, more usefull when only padding a single axis

    # Arguments:
        array: the array to pad
        new_size: the new size of the specified axis
        axis: axis along which to pad
        pad_value: pad value,
        pad_right: boolean, pad on the right or left side

    # Returns:
        padded_array: np.array

    '''
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(axis, array.shape[axis], new_size)
    pad_width = [(0,0)]*len(array.shape)

    #pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)


def tensorise_smiles(smiles, max_degree=5, max_atoms=None):
    """
    Takes a list of smiles and turns the graphs in tensor representation.

    # Arguments:
        molecules: chemml.chem.Molecule object or list

        smiles: a list (or iterable) of smiles representations
        max_atoms: the maximum number of atoms per molecule (to which all
            molecules will be padded), use `None` for auto
        max_degree: max_atoms: the maximum number of neigbour per atom that each
            molecule can have (to which all molecules will be padded), use `None`
            for auto

        **NOTE**: It is not recommended to set max_degree to `None`/auto when
            using `NeuralGraph` layers. Max_degree determines the number of
            trainable parameters and is essentially a hyperparameter.
            While models can be rebuilt using different `max_atoms`, they cannot
            be rebuild for different values of `max_degree`, as the architecture
            will be different.

            For organic molecules `max_degree=5` is a good value (Duvenaud et. al, 2015)


    # Returns:
        atoms: np.array, An atom feature np.array of size `(molecules, max_atoms, atom_features)`
        bonds: np.array, A bonds np.array of size `(molecules, max_atoms, max_neighbours)`
        edges: np.array, A connectivity array of size `(molecules, max_atoms, max_neighbours, bond_features)`
    TODO:
        * Arguments for sparse vector encoding

    """

    # import sizes
    n = len(smiles)
    n_atom_features = num_atom_features()
    n_bond_features = num_bond_features()

    # preallocate atom tensor with 0's and bond tensor with -1 (because of 0 index)
    # If max_degree or max_atoms is set to None (auto), initialise dim as small
    #   as possible (1)
    atom_tensor = np.zeros((n, max_atoms or 1, n_atom_features))
    bond_tensor = np.zeros((n, max_atoms or 1, max_degree or 1, n_bond_features))
    edge_tensor = -np.ones((n, max_atoms or 1, max_degree or 1), dtype=int)

    for mol_ix, s in enumerate(smiles):

        #load mol, atoms and bonds
        mol = Chem.MolFromSmiles(s)
        assert mol is not None, 'Could not parse smiles {}'.format(s)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # If max_atoms is exceeded, resize if max_atoms=None (auto), else raise
        if len(atoms) > atom_tensor.shape[1]:
            assert max_atoms is None, 'too many atoms ({0}) in molecule: {1}'.format(len(atoms), s)
            atom_tensor = padaxis(atom_tensor, len(atoms), axis=1)
            bond_tensor = padaxis(bond_tensor, len(atoms), axis=1)
            edge_tensor = padaxis(edge_tensor, len(atoms), axis=1, pad_value=-1)

        rdkit_ix_lookup = {}
        connectivity_mat = {}

        for atom_ix, atom in enumerate(atoms):
            # write atom features
            atom_tensor[mol_ix, atom_ix, : n_atom_features] = atom_features(atom)

            # store entry in idx
            rdkit_ix_lookup[atom.GetIdx()] = atom_ix

        # preallocate array with neighbour lists (indexed by atom)
        connectivity_mat = [ [] for _ in atoms]

        for bond in bonds:
            # lookup atom ids
            a1_ix = rdkit_ix_lookup[bond.GetBeginAtom().GetIdx()]
            a2_ix = rdkit_ix_lookup[bond.GetEndAtom().GetIdx()]

            # lookup how many neighbours are encoded yet
            a1_neigh = len(connectivity_mat[a1_ix])
            a2_neigh = len(connectivity_mat[a2_ix])

            # If max_degree is exceeded, resize if max_degree=None (auto), else raise
            new_degree = max(a1_neigh, a2_neigh) + 1
            if new_degree > bond_tensor.shape[2]:
                assert max_degree is None, 'too many neighours ({0}) in molecule: {1}'.format(new_degree, s)
                bond_tensor = padaxis(bond_tensor, new_degree, axis=2)
                edge_tensor = padaxis(edge_tensor, new_degree, axis=2, pad_value=-1)

            # store bond features
            bond_features = np.array(bond_features(bond), dtype=int)
            bond_tensor[mol_ix, a1_ix, a1_neigh, :] = bond_features
            bond_tensor[mol_ix, a2_ix, a2_neigh, :] = bond_features

            #add to connectivity matrix
            connectivity_mat[a1_ix].append(a2_ix)
            connectivity_mat[a2_ix].append(a1_ix)

        #store connectivity matrix
        for a1_ix, neighbours in enumerate(connectivity_mat):
            degree = len(neighbours)
            edge_tensor[mol_ix, a1_ix, : degree] = neighbours

    return atom_tensor, bond_tensor, edge_tensor


def concat_mol_tensors(mol_tensors_list, match_degree=True, match_max_atoms=False):
    '''Concatenates a list of molecule tensors

    # Arguments:
        mol_tensor_list: list of molecule tensors (e.g. list of
        `(atoms, bonds, edges)`-triplets)
        match_degree: bool, if True, the degrees of all tensors should match,
            if False, unmatching degrees will be padded to align them.
        match_max_atoms: bool, simular to match_degree but for max_atoms

    # Retuns:
        a single molecule tensor (as returned by `tensorise_smiles`)
    '''

    assert isinstance(mol_tensors_list, (tuple, list)), 'Provide a list or tuple of molecule tensors to concatenate'

    # get max_atoms (#1) of atoms (#0) tensor of first batch (#0)
    # and max_degree (#2) of bonds (#1) tensor of first batch (#0)
    max_atoms = mol_tensors_list[0][0].shape[1]
    max_degree = mol_tensors_list[0][1].shape[2]


    # Obtain the max_degree and max_atoms of all tensors in the list
    for atoms, bonds, edges in mol_tensors_list:
        assert bonds.shape[0] == edges.shape[0] == atoms.shape[0], "batchsize doesn't match within tensor"
        assert bonds.shape[1] == edges.shape[1] == atoms.shape[1], "max_atoms doesn't match within tensor"
        assert bonds.shape[2] == edges.shape[2], "degree doesn't match within tensor"

        if match_max_atoms:
            assert max_atoms == atoms.shape[1], '`max_atoms` of molecule tensors does not match, set `match_max_atoms` to False to adjust'
        else:
            max_atoms = max(max_atoms, atoms.shape[1])

        if match_degree:
            assert max_degree == bonds.shape[2], '`degree` of molecule tensors does not match, set `match_degree` to False to adjust'
        else:
            max_degree = max(max_degree,  bonds.shape[2])

    # Pad if necessary and separate tensors
    atoms_list = []
    bonds_list = []
    edges_list = []
    for atoms, bonds, edges in mol_tensors_list:

        atoms = padaxis(atoms, max_atoms, axis=1)
        bonds = padaxis(bonds, max_atoms, axis=1)
        edges = padaxis(edges, max_atoms, axis=1, pad_value=-1)

        bonds = padaxis(bonds, max_degree, axis=2)
        edges = padaxis(edges, max_degree, axis=2, pad_value=-1)

        atoms_list.append(atoms)
        bonds_list.append(bonds)
        edges_list.append(edges)

    #stack along batch-size axis
    atoms = np.concatenate(atoms_list, axis=0)
    bonds = np.concatenate(bonds_list, axis=0)
    edges = np.concatenate(edges_list, axis=0)

    return atoms, bonds, edges


def tensorise_smiles_mp(smiles, max_degree=5, max_atoms=None, workers=cpu_count()-1, chunksize=3000, verbose=True):
    ''' Multiprocess implementation of `tensorise_smiles`

    # Arguments:
        See `tensorise_smiles` documentation

    # Additional arguments:
        workers: int, num parallel processes
        chunksize: int, num molecules tensorised per worker, bigger chunksize is
            preffered as each process will preallocate np.arrays

    # Returns:
        See `tensorise_smiles` documentation

    # TODO:
        - fix python keyboardinterrupt bug:
          https://noswap.com/blog/python-multiprocessing-keyboardinterrupt
        - replace progbar with proper logging
    '''

    pool = Pool(processes=workers)

    # Create an iterator
    #http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    smiles_chunks = chunks(smiles, chunksize)

    # MAP: Tensorise in parallel
    map_function = partial(tensorise_smiles, max_degree=max_degree, max_atoms=max_atoms)
    if verbose:
        print('Tensorising molecules in batches...')
        pbar = Progbar(len(smiles), width=50)
        tensor_list = []
        for tensors in pool.imap(map_function, smiles_chunks):
            pbar.add(tensors[0].shape[0])
            tensor_list.append(tensors)
        print('Merging batch tensors...    ', end='')
    else:
        tensor_list = pool.map(map_function, smiles_chunks)
    if verbose:
        print('[DONE]')

    # REDUCE: Concatenate the obtained tensors
    pool.close()
    pool.join()
    return concat_mol_tensors(tensor_list, match_degree=max_degree!=None, match_max_atoms=max_atoms!=None)