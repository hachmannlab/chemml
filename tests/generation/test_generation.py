import os
import random

import pytest

pytest.importorskip("openbabel")

from openbabel import pybel

from chemml.generation.assembly import (
    check_building_blocks,
    create_fused,
    create_link,
    if_add,
    lipinski,
    molecule,
)
from chemml.generation.ga import GeneticAlgorithm, count_list, nested_lookup

BLOCKS = ['C', 'S', 'O', 'c1ccccc1', 'c1nnc(s1)', 'c1ccc2c(c1)ccc(c2)']
NO_BB = {'bb_final_lib': False}


def objective(mol_ob):
    return random.random()


def make_ga(seed=0, output_dir=None, **kw):
    mols = [molecule(s, 'F' + str(i + 1)) for i, s in enumerate(BLOCKS)]
    args = dict(evaluate=objective, fitness=(('max', 0.01),), crossover_size=6, mutation_size=4,
                algorithm=2, initial_mols=mols, rules_dict={'bb_final_lib': False},
                output_dir=output_dir, seed=seed)
    args.update(kw)
    return GeneticAlgorithm(**args)


def _rules(**kw):
    return dict(kw, **NO_BB)


def test_create_fused_substitutions():
    mol1 = molecule('C1CCCCC1', 'F1-F2')
    mol2 = molecule('N1CC(Cl)CC1', 'F3')
    lib = create_fused(mol1, mol2, {'bb_final_lib': False})
    assert isinstance(lib, list)
    assert len(lib) >= 4


def test_create_fused_general():
    mol1 = molecule('CCc1ccccc1', 'F1-F2')
    mol2 = molecule('o1cccc1', 'F3')
    lib = create_fused(mol1, mol2, {'bb_final_lib': False})
    assert isinstance(lib, list)
    assert len(lib) >= 4


def test_create_link_general():
    mol1 = molecule('CCc1ccccc1', 'F1-F2-F5')
    mol2 = molecule('CC', 'F3')
    lib = create_link(mol1, mol2, {'bb_final_lib': False})
    assert isinstance(lib, list)
    assert len(lib) == 5


def test_count_list_and_nested_lookup():
    indi = ['F1', [], ['F2', ['C'], []]]
    empty, filled = count_list(indi)
    assert empty == [[1], [2, 2]]
    assert filled == [[2]]
    assert nested_lookup(indi, [2, 2]) == []
    assert nested_lookup(indi, [2]) == ['F2', ['C'], []]


@pytest.mark.parametrize('algorithm', [1, 2, 3, 4])
def test_search(tmp_path, algorithm):
    ga = make_ga(output_dir=str(tmp_path), algorithm=algorithm)
    ga.search(n_generations=1)
    assert len(ga.population) >= 10
    assert os.path.exists(tmp_path / 'generation_1.csv')


def test_resumed_search_writes_additional_generation(tmp_path):
    ga = make_ga(output_dir=str(tmp_path))
    ga.search(n_generations=1)
    generation_one = (tmp_path / 'generation_1.csv').read_bytes()

    ga.search(n_generations=1)

    assert (tmp_path / 'generation_1.csv').read_bytes() == generation_one
    assert os.path.exists(tmp_path / 'generation_2.csv')


def test_chromosomes_build_molecules():
    ga = make_ga()
    for indi in ga.pop_generator(5):
        assert ga.list_to_smiles(list(indi))


def test_operators_return_individuals():
    ga = make_ga()
    c1, c2 = ga.pop_generator(2)
    assert all(isinstance(i, tuple) for i in ga.crossover(c1, c2))
    assert isinstance(ga.custom_mutate(c1), tuple)


def test_seed_is_reproducible():
    a = make_ga(seed=3).pop_generator(5)
    b = make_ga(seed=3).pop_generator(5)
    assert a == b


def test_batch(tmp_path):
    ga = make_ga(output_dir=str(tmp_path))
    ga.batch(())
    assert ga.population is None
    assert os.path.exists(tmp_path / 'new_molecules.csv')


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_ga(fitness=(('max', 0),))
    with pytest.raises(ValueError):
        make_ga().search(init_ratio=0.6, crossover_ratio=0.5)


def test_if_add_general():
    assert if_add('CC', _rules(include_bb=['C']), 'F1-F3-F2')
    assert if_add('CCCC', _rules(**{'2': (0, 5)}), 'F1')
    assert if_add('CC', _rules(**{'3': (0, 2)}), 'F1')
    assert if_add('CO', _rules(**{'4': (25, 35)}), 'F1')
    assert if_add('C1CCC1', _rules(**{'5': (0, 2)}), 'F1')
    assert if_add('c1ccc1', _rules(**{'6': (0, 2)}), 'F1')
    assert if_add('C1CCC1', _rules(**{'7': (0, 2)}), 'F1')
    assert if_add('CC', _rules(**{'8': (0, 1)}), 'F1')
    assert if_add('CC=CC', _rules(**{'9': (0, 2)}), 'F1')
    assert if_add('CC#CC', _rules(**{'10': (0, 2)}), 'F1')
    assert if_add('CCOCOCCC', _rules(heteroatoms=[('O', 2)]), 'F1')


def test_rules_run_without_error():
    # Outcomes of these rules were not asserted in the original ChemLG tests.
    if_add('CCCC', _rules(lipinski=True), 'F1')
    if_add('CCCC', _rules(fingerprint=[['c1ccccc1', 0.1]]), 'F1')
    if_add('CCCC', _rules(**{'14': ['CCCCCC']}), 'F1')
    if_add('CCCC', _rules(**{'15': ['CCCCCCC']}), 'F1')


def test_if_add_exceptions():
    assert not if_add('', NO_BB, 'F1')
    assert not if_add('smiles', NO_BB, 'F1')
    assert not if_add('CC', _rules(include_bb=['O']), 'F1-F3')
    assert not if_add('CC', _rules(**{'2': (10, 20)}), 'F1', check_min=True)
    assert not if_add('CC', _rules(**{'3': (10, 20)}), 'F1', check_min=True)
    assert not if_add('CCCCCC', _rules(**{'4': (10, 20)}), 'F1', check_min=True)
    assert not if_add('C1CCC1', _rules(**{'5': (10, 20)}), 'F1', check_min=True)
    assert not if_add('C1CCC1', _rules(**{'6': (10, 20)}), 'F1', check_min=True)
    assert not if_add('c1ccc1', _rules(**{'7': (10, 20)}), 'F1', check_min=True)
    assert not if_add('CCCC', _rules(**{'8': (40, 50)}), 'F1', check_min=True)
    assert not if_add('CC=CC', _rules(**{'9': (40, 50)}), 'F1', check_min=True)
    assert not if_add('CC#CC', _rules(**{'10': (40, 50)}), 'F1', check_min=True)
    assert not if_add('CCOCOCCCO', _rules(heteroatoms=[('O', 2)]), 'F1', check_min=True)


def test_lipinski():
    descr = lipinski(pybel.readstring("smi", 'CCC'))
    assert isinstance(descr, dict)
    assert set(descr) == {'molwt', 'HBD', 'HBA', 'logP'}


def test_molecule_general():
    mol = molecule('CcC', 'F1-F2')
    assert isinstance(mol, dict)
    assert mol['can_smiles'] == 'C[CH]C\t\n'
    assert mol['code'] == 'F1-F2'
    assert mol['smiles'] == 'CcC'
    assert mol['reverse_smiles'] == 'C[CH]C'


def test_check_building_blocks():
    assert check_building_blocks('CCO') == 'CCO'
    # assert check_building_blocks('1/C2H6O/c1-2-3/h3H,2H2,1H3') != ''
    with pytest.raises(ValueError):
        check_building_blocks('not a molecule!!')