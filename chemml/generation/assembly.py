"""
Molecule assembly primitives (Open Babel based) ported from ChemLG's libgen module.

Molecules are handled as dictionaries with the keys ``smiles``, ``code``, ``can_smiles`` and
``reverse_smiles`` (see :func:`molecule`). Reactive sites are encoded in ``reverse_smiles``:
hydrogens are the attachment points unless the user marked explicit handles with ``[Ra]``
(or ``[X]``) atoms.

Ported from ChemLG (C) 2015-2020 Johannes Hachmann, Mohammad Atif Faiz Afzal; BSD 3-Clause License.
"""

import logging
from collections import defaultdict

from openbabel import pybel
from openbabel.openbabel import OBAtomAtomIter

logger = logging.getLogger(__name__)

_FR, _RA = 87, 88
# the parent molecule must stay referenced or the atom pointer dangles
_FR_MOL = pybel.readstring('smi', "[Fr]")
_FR_ATOM = _FR_MOL.OBMol.GetAtom(1)


def molecule(smiles, code):
    """
    Create the dictionary describing a molecule.

    Parameters
    ----------
    smiles: str
        smiles string of molecule
    code: str
        code for molecule (F1, F2 ...)

    Returns
    -------
    mol: dict
        dictionary with keys - smiles, code, can_smiles, reverse_smiles
    """
    mol = {'smiles': smiles, 'code': code}
    obm = pybel.readstring("smi", smiles)
    mol['can_smiles'] = obm.write("can").replace('\r', '')
    obm_can = pybel.readstring("smi", mol['can_smiles'])
    mol['reverse_smiles'] = reverse_mol(obm_can, list(obm_can.atoms)).replace('\t', '')
    return mol


def reverse_mol(mol, atoms):
    """
    Convert a molecule's potential reactive sites into H atoms.

    If the molecule contains [Ra] handles, all existing H atoms are replaced by Fr atoms (so they are
    no longer considered sites) and the Ra atoms are converted to H atoms.

    Parameters
    ----------
    mol: object
        Open Babel (pybel) object of a molecule
    atoms: list
        list of atoms in the molecule

    Returns
    -------
    smiles: str
        canonical smiles of transformed molecule
    """
    atom_num = [atom.OBAtom.GetAtomicNum() for atom in atoms]
    Fratom = _FR_ATOM

    if _RA in atom_num:
        for atom in atoms:
            hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()
            index = atom.OBAtom.GetIdx()

            while hcount != 0:
                size = len(list(mol.atoms))
                mol.OBMol.InsertAtom(Fratom)
                mol.OBMol.AddBond(index, size + 1, 1, 0, -1)
                x = mol.OBMol.GetAtom(index).GetImplicitHCount() - 1
                if x >= 0:
                    mol.OBMol.GetAtom(index).SetImplicitHCount(x)
                hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()

        for atom in list(mol.atoms):
            if atom.OBAtom.GetAtomicNum() == _RA:
                atom.OBAtom.SetAtomicNum(1)

    return mol.write("can")[:-2]


def check_building_blocks(smiles):
    """
    Validate a building block given as SMILES or InChI.

    Parameters
    ----------
    smiles: str
        molecule representation (smiles/inchi)

    Returns
    -------
    smiles: str
        smiles representation of the molecule

    Raises
    ------
    ValueError
        if the string is neither a valid SMILES nor a valid InChI.
    """
    try:
        return str(pybel.readstring("inchi", smiles)).strip()
    except Exception:
        pass
    try:
        pybel.readstring("smi", smiles)
    except Exception:
        raise ValueError("The SMILES/InChI string '{}' is not valid.".format(smiles))
    return smiles


def lipinski(mol):
    """
    Return the values of the Lipinski descriptors.

    Parameters
    ----------
    mol: object
        pybel object of the molecule

    Returns
    -------
    desc: dict
        molwt, HBD, HBA and logP
    """
    HBD = pybel.Smarts("[#7,#8;!H0]")
    HBA = pybel.Smarts("[#7,#8]")
    return {'molwt': mol.molwt,
            'HBD': len(HBD.findall(mol)),
            'HBA': len(HBA.findall(mol)),
            'logP': mol.calcdesc(['logP'])['logP']}


def unique_structs(mol, smarts):
    """
    Number of unique matches of a SMARTS sub-structure in a molecule.

    Parameters
    ----------
    mol: object
        pybel object of molecule
    smarts: str
        smarts representation of sub-structure

    Returns
    -------
    num_unique_matches: int
    """
    smarts = pybel.Smarts(smarts)
    smarts.obsmarts.Match(mol.OBMol)
    return len(smarts.findall(mol))


def if_add(molc, rules, code, check_min=False):
    """
    Determine if a molecule satisfies the generation rules.

    Parameters
    ----------
    molc: str
        reversed canonical smiles of molecule whose validity is to be determined
    rules: dict
        dictionary of rules. Keys: 'include_bb', '2' (bonds), '3' (atoms), '4' (mol. weight),
        '5' (rings), '6' (aromatic rings), '7' (non-aromatic rings), '8' (single bonds),
        '9' (double bonds), '10' (triple bonds), 'heteroatoms', 'lipinski', 'fingerprint',
        '14' (substructures to include), '15' (substructures to exclude). Range rules are
        (min, max) tuples; the minima are only checked if ``check_min`` is True.
    code: str
        code for molecule (F1, F2 ...)
    check_min: bool
        also enforce the minimum values of range rules

    Returns
    -------
    add: bool
    """
    if molc == '':
        return False
    try:
        pybel.readstring("smi", molc)
    except Exception:
        logger.warning("invalid smiles: %s", molc)
        return False

    mol = molecule(molc, code)
    mol_ob = pybel.readstring("smi", mol['reverse_smiles'])

    for atom in list(mol_ob.atoms):
        if atom.OBAtom.GetAtomicNum() in (_FR, _RA):
            atom.OBAtom.SetAtomicNum(1)

    if 'include_bb' in rules:
        for item in rules['include_bb']:
            if unique_structs(mol_ob, item) <= 0:
                return False

    if '2' in rules and not mol_ob.OBMol.NumBonds() <= rules['2'][1]:
        return False

    if '3' in rules and not len(list(mol_ob.atoms)) <= rules['3'][1]:
        return False

    if '5' in rules and not len(mol_ob.OBMol.GetSSSR()) <= rules['5'][1]:
        return False

    no_ar, no_non_ar = 0, 0
    if '6' in rules or '7' in rules:
        for r in mol_ob.OBMol.GetSSSR():
            if r.IsAromatic():
                no_ar += 1
            else:
                no_non_ar += 1
        if '6' in rules and not no_ar <= rules['6'][1]:
            return False
        if '7' in rules and not no_non_ar <= rules['7'][1]:
            return False

    if '8' in rules and not unique_structs(mol_ob, "*-*") <= rules['8'][1]:
        return False

    if '9' in rules and not unique_structs(mol_ob, "*=*") <= rules['9'][1]:
        return False

    if '10' in rules and not unique_structs(mol_ob, "*#*") <= rules['10'][1]:
        return False

    if 'heteroatoms' in rules:
        for item in rules['heteroatoms']:
            if item[0] in ('C', 'S', 'N', 'O', 'c', 's', 'n', 'o'):
                no_at = unique_structs(mol_ob, item[0].lower()) + unique_structs(mol_ob, item[0].upper())
            else:
                no_at = unique_structs(mol_ob, item[0])
            if no_at > item[1]:
                return False

    if 'lipinski' in rules:
        d = lipinski(mol_ob)
        if not (d['molwt'] <= 500 and d['HBD'] <= 5 and d['HBA'] <= 10 and d['logP'] <= 5):
            return False

    if 'fingerprint' in rules:
        for mol_to_comp in rules['fingerprint']:
            mol2 = pybel.readstring('smi', mol_to_comp[0])
            if (mol_ob.calcfp() | mol2.calcfp()) <= float(mol_to_comp[1]):
                return False

    if '14' in rules:
        for item in rules['14']:
            if unique_structs(mol_ob, item) == 0:
                return False

    if '15' in rules:
        for item in rules['15']:
            if unique_structs(mol_ob, item) > 0:
                return False

    if check_min:
        if '2' in rules and not rules['2'][0] <= mol_ob.OBMol.NumBonds():
            return False
        if '3' in rules and not rules['3'][0] <= len(list(mol_ob.atoms)):
            return False
        if '4' in rules and not rules['4'][0] <= int(mol_ob.OBMol.GetMolWt()) <= rules['4'][1]:
            return False
        if '5' in rules and not rules['5'][0] <= len(mol_ob.OBMol.GetSSSR()):
            return False
        if '6' in rules and not rules['6'][0] <= no_ar:
            return False
        if '7' in rules and not rules['7'][0] <= no_non_ar:
            return False
        if '8' in rules and not rules['8'][0] <= unique_structs(mol_ob, "*-*"):
            return False
        if '9' in rules and not rules['9'][0] <= unique_structs(mol_ob, "*=*"):
            return False
        if '10' in rules and not rules['10'][0] <= unique_structs(mol_ob, "*#*"):
            return False

    return True


def _bond(mol_combi, i, j):
    """Single bond between atoms i and j, consuming one implicit H on each."""
    mol_combi.OBMol.AddBond(i, j, 1, 0, -1)
    x = mol_combi.OBMol.GetAtom(i).GetImplicitHCount() - 1
    y = mol_combi.OBMol.GetAtom(j).GetImplicitHCount() - 1
    if x >= 0:
        mol_combi.OBMol.GetAtom(i).SetImplicitHCount(x)
    if y >= 0:
        mol_combi.OBMol.GetAtom(j).SetImplicitHCount(y)


def get_index_list(mol, atoms):
    """
    Indices of atoms of a molecule that can be reacted, without symmetry-duplicate sites.

    Parameters
    ----------
    mol: dict
        molecule dictionary
    atoms: list
        list of atoms in the molecule

    Returns
    -------
    atoms_index: list
    """
    can_smiles_list, atoms_index = [], []
    Fratom = _FR_ATOM

    for atom in atoms:
        hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()
        if hcount == 0:
            continue

        newmol = pybel.readstring("smi", mol['reverse_smiles'])
        newmol.OBMol.InsertAtom(Fratom)
        index = atom.OBAtom.GetIdx()
        newmol.OBMol.AddBond(index, len(atoms) + 1, 1, 0, -1)
        x = newmol.OBMol.GetAtom(index).GetImplicitHCount() - 1
        if x >= 0:
            newmol.OBMol.GetAtom(index).SetImplicitHCount(x)
        # a canonical smiles with the Fr marker identifies symmetry-equivalent sites
        can_smiles = newmol.write("can")

        if can_smiles not in can_smiles_list:
            can_smiles_list.append(can_smiles)
            atoms_index.append(index)

    return atoms_index


def create_link(mol1, mol2, rules):
    """
    Create all possible links between two molecules.

    Parameters
    ----------
    mol1, mol2: dict
        molecule dictionaries
    rules: dict
        generation rules (see :func:`if_add`)

    Returns
    -------
    library_two: list
        list of molecule dictionaries
    """
    library_two = []
    mol1_ob = pybel.readstring("smi", mol1['reverse_smiles'])
    mol2_ob = pybel.readstring("smi", mol2['reverse_smiles'])
    mol1_index_list = get_index_list(mol1, list(mol1_ob.atoms))
    mol2_index_list = get_index_list(mol2, list(mol2_ob.atoms))

    smiles_combi = mol1['reverse_smiles'] + '.' + mol2['reverse_smiles']
    code = mol1['code'] + '-' + mol2['code']

    for index1 in mol1_index_list:
        for index2 in mol2_index_list:
            mol_combi = pybel.readstring("smi", smiles_combi)
            _bond(mol_combi, index1, index2 + len(mol1_ob.atoms))
            can_mol_combi = mol_combi.write("can")
            if if_add(can_mol_combi, rules, code):
                library_two.append(molecule(can_mol_combi, code))
    return library_two


def get_fusion_index(molc, mol_type):
    """
    Potential fusion sites of a molecule after removing duplicate sites.

    Parameters
    ----------
    molc: dict
        molecule dictionary
    mol_type: int
        1: molecule that keeps all its atoms; 2: molecule that loses the two fusing atoms.

    Returns
    -------
    atom_pair_list: list
        mol_type 1 - [atom, alpha atom]; mol_type 2 - [atom, 1st alpha, 2nd alpha, beta of 1st alpha]
    n_atoms: int
        number of atoms in the molecule
    substitutions: dict
        side-chain substitutions on the atoms that will fuse
    """
    Fratom = _FR_ATOM

    def modify_smiles(a1, a2):
        newmol = pybel.readstring("smi", molc['reverse_smiles'])
        newmol.OBMol.InsertAtom(Fratom)
        newmol.OBMol.InsertAtom(Fratom)
        newmol.OBMol.AddBond(a1, len(atoms) + 1, 1, 0, -1)
        x = newmol.OBMol.GetAtom(a1).GetImplicitHCount() - 1
        if x >= 0:
            newmol.OBMol.GetAtom(a1).SetImplicitHCount(x)
        newmol.OBMol.AddBond(a2, len(atoms) + 2, 1, 0, -1)
        x = newmol.OBMol.GetAtom(a2).GetImplicitHCount() - 1
        if x >= 0:
            newmol.OBMol.GetAtom(a2).SetImplicitHCount(x)
        return newmol.write("can")

    mol = pybel.readstring('smi', molc['reverse_smiles'])
    atoms = list(mol.atoms)
    atom_pair_list, can_list, substitutions = [], [], defaultdict(list)
    for atom in atoms:
        index = atom.OBAtom.GetIdx()
        hcount = atom.OBAtom.ExplicitHydrogenCount() + atom.OBAtom.GetImplicitHCount()
        if hcount == 0 or not mol.OBMol.GetAtom(index).IsInRing():
            continue
        for neighbor in OBAtomAtomIter(mol.OBMol.GetAtom(index)):
            neigbor_hcount = neighbor.ExplicitHydrogenCount() + neighbor.GetImplicitHCount()
            alpha_1 = neighbor.GetIdx()
            if neigbor_hcount == 0 or not neighbor.IsInRing():
                if not neighbor.IsInRing():
                    if alpha_1 not in substitutions[index]:
                        substitutions[index].append(alpha_1)
                continue

            if mol_type == 1:
                modified = modify_smiles(index, alpha_1)
                if modified not in can_list:
                    can_list.append(modified)
                    atom_pair_list.append([index, alpha_1])

            if mol_type == 2:
                for second_neighbor in OBAtomAtomIter(neighbor):
                    beta_1 = second_neighbor.GetIdx()
                    if not second_neighbor.IsInRing() or second_neighbor.GetIdx() == index:
                        if not second_neighbor.IsInRing():
                            if beta_1 not in substitutions[alpha_1]:
                                substitutions[alpha_1].append(beta_1)
                        continue
                    break
                for second_neighbor in OBAtomAtomIter(mol.OBMol.GetAtom(index)):
                    if not second_neighbor.IsInRing() or second_neighbor.GetIdx() == alpha_1:
                        continue
                    alpha_2 = second_neighbor.GetIdx()
                    break
                modified = modify_smiles(index, alpha_1)
                if modified not in can_list:
                    can_list.append(modified)
                    atom_pair_list.append([index, alpha_1, alpha_2, beta_1])

    return atom_pair_list, len(atoms), substitutions


def create_fused(mol1, mol2, rules):
    """
    All molecules resulting from fusion of two molecules.

    Parameters
    ----------
    mol1, mol2: dict
        molecule dictionaries
    rules: dict
        generation rules (see :func:`if_add`)

    Returns
    -------
    library_two: list
        list of molecule dictionaries
    """
    mol1_index_list, size1, subs_1 = get_fusion_index(mol1, 1)
    mol2_index_list, size2, subs_2 = get_fusion_index(mol2, 2)
    smiles_combi = mol1['reverse_smiles'] + '.' + mol2['reverse_smiles']
    library_two = []
    code = mol1['code'] + ':' + mol2['code']
    for item1 in mol1_index_list:
        for item2 in mol2_index_list:
            prime_subs, alpha_subs = subs_1[item1[0]] + subs_2[item2[0]], subs_1[item1[1]] + subs_2[item2[1]]
            mol_combi = pybel.readstring("smi", smiles_combi)
            prime_atom = mol_combi.OBMol.GetAtom(item1[0])
            alpha_1_atom = mol_combi.OBMol.GetAtom(item1[1])
            alpha_2_atom = mol_combi.OBMol.GetAtom(size1 + item2[2])
            beta_1_atom = mol_combi.OBMol.GetAtom(size1 + item2[3])

            # fusing atoms of different elements: keep the non-carbon one
            if prime_atom.GetAtomicNum() != mol_combi.OBMol.GetAtom(item2[0] + size1).GetAtomicNum():
                prime_atom.SetAtomicNum(list(set([prime_atom.GetAtomicNum(), mol_combi.OBMol.GetAtom(item2[0] + size1).GetAtomicNum()]) - set([6]))[0])

            if alpha_1_atom.GetAtomicNum() != mol_combi.OBMol.GetAtom(item2[1] + size1).GetAtomicNum():
                alpha_1_atom.SetAtomicNum(list(set([alpha_1_atom.GetAtomicNum(), mol_combi.OBMol.GetAtom(item2[1] + size1).GetAtomicNum()]) - set([6]))[0])

            first_mol_aromatic = alpha_1_atom.IsAromatic() and prime_atom.IsAromatic()
            second_mol_aromatic = alpha_2_atom.IsAromatic() and beta_1_atom.IsAromatic()

            # valency check of the fusing atoms given their side chains
            if (first_mol_aromatic or second_mol_aromatic) and (len(prime_subs) + len(alpha_subs) > 0):
                return library_two
            if not (first_mol_aromatic or second_mol_aromatic) and (len(prime_subs) > 1 or len(alpha_subs) > 1):
                return library_two

            _bond(mol_combi, item1[0], item2[2] + size1)
            _bond(mol_combi, item1[1], item2[3] + size1)

            for subs_ind in subs_2[item2[0]]:
                _bond(mol_combi, item1[0], subs_ind + size1)
            for subs_ind in subs_2[item2[1]]:
                _bond(mol_combi, item1[1], subs_ind + size1)

            for i in sorted([size1 + item2[0], size1 + item2[1]], reverse=True):
                mol_combi.OBMol.DeleteAtom(mol_combi.OBMol.GetAtom(i))

            if second_mol_aromatic:
                for mol1_atom in [mol_combi.OBMol.GetAtom(item1[0]), mol_combi.OBMol.GetAtom(item1[1])]:
                    mol1_atom.SetAromatic()
                for atom in list(mol_combi.atoms)[size1:]:
                    mol_combi.OBMol.GetAtom(atom.OBAtom.GetIdx()).SetAromatic()

            can_mol_combi = mol_combi.write("can")
            if if_add(can_mol_combi, rules, code):
                library_two.append(molecule(can_mol_combi, code))

    return library_two
