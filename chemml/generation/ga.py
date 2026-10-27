"""
Genetic algorithm for molecule generation, ported from ChemLG's libgen module.

Individuals are variable-size assembly trees stored as nested lists. Each building block is a
list ``[block_code, handle_1, handle_2, ...]`` where each handle is either ``[]`` (free site),
``['C']`` (the site through which the block is bonded to its parent) or a nested block. This
representation does not fit the fixed-length chromosomes of ``chemml.optimization.GeneticAlgorithm``,
hence this class is self-contained.

Ported from ChemLG (C) 2015-2020 Johannes Hachmann, Mohammad Atif Faiz Afzal, Gaurav Vishwakarma;
BSD 3-Clause License.
"""

import logging
import os
import random
from copy import deepcopy

import pandas as pd
from openbabel import pybel

from .assembly import get_index_list, if_add

logger = logging.getLogger(__name__)


class building_blocks(object):
    """
    A building block prepared for the genetic algorithm.

    Parameters
    ----------
    mol: dict
        dictionary created by :func:`chemml.generation.assembly.molecule`

    Attributes
    ----------
    smiles: str
        reverse smiles (reactive sites are H atoms)
    smiles_struct: str
        molecule code
    atom_len: int
        number of atoms
    index_list: list
        indices of the unique reactive sites
    spaces: int
        number of unique reactive sites
    """

    def __init__(self, mol):
        self.smiles = mol['reverse_smiles']
        self.smiles_struct = mol['code']
        mol_ob = pybel.readstring("smi", mol['reverse_smiles'])
        self.atom_len = len(list(mol_ob.atoms))
        self.index_list = get_index_list(mol, list(mol_ob.atoms))
        self.spaces = len(self.index_list)


class GeneticAlgorithm(object):
    """
    Genetic algorithm that assembles molecules from building blocks and optimizes an objective function.

    Parameters
    ----------
    evaluate: function
        Objective function. Its first argument is a pybel object of a molecule. It returns a number or
        a tuple of numbers (the target properties).

    fitness: tuple of tuple(s)
        One tuple per target property: (optima, cutoff), e.g. (('max', 5.6), ('min', 20)). For 'max' the
        cutoff is the lower acceptable limit, for 'min' the maximum allowed value. Cutoffs cannot be zero.

    crossover_size: int
        size of the crossover population

    mutation_size: int
        size of the mutation population. The sum of both is the number of new individuals per generation.

    algorithm: int
        Selection algorithm (1-4), see :meth:`search`.

    initial_mols: list
        building blocks, as dictionaries from :func:`chemml.generation.assembly.molecule`

    rules_dict: dict
        generation rules, see :func:`chemml.generation.assembly.if_add`

    output_dir: str, optional (default=None)
        If given, each generation is written to ``<output_dir>/generation_<n>.csv`` and batch mode writes
        ``new_molecules.csv`` there.

    seed: int, optional (default=None)
        Seed for a private random number generator. If None, the global ``random`` module is used.
    """

    def __init__(self, evaluate, fitness, crossover_size, mutation_size, algorithm,
                 initial_mols, rules_dict, output_dir=None, seed=None):
        self.evaluate = evaluate
        self.algo = algorithm
        self.population, self.fit_list = None, ()
        self.crossover_size = int(crossover_size)
        self.mutation_size = int(mutation_size)
        self.fit_val = []
        self.pop_size = self.crossover_size + self.mutation_size
        self.output_dir = output_dir
        self._rng = random if seed is None else random.Random(seed)

        for i in fitness:
            if i[1] == 0:
                raise ValueError("Cutoff values in the fitness cannot be zero.")
            self.fit_val.append((1, i[1]) if i[0].lower() == 'max' else (-1, i[1]))

        self.bb = [building_blocks(i) for i in initial_mols]
        self.rules_dict = rules_dict

    def pop_generator(self, n):
        return [tuple(self.chromosome_generator()) for _ in range(n)]

    def chromosome_generator(self):
        """Generate a random individual of 2-5 building blocks."""
        i = 0
        chromosome = []
        ind_len = self._rng.randint(2, 5)
        while i < ind_len:
            if i == 0:
                r = self._rng.randint(0, len(self.bb) - 1)
                chromosome.append(self.bb[r].smiles_struct)
                for _ in range(self.bb[r].spaces):
                    chromosome.append([])
            else:
                avl_pos = count_list(chromosome)[0]
                if len(avl_pos) <= 0:
                    return chromosome
                r = self._rng.randint(0, len(avl_pos) - 1)             # free handle of the parent
                s = self._rng.randint(0, len(self.bb) - 1)             # new block
                t = self._rng.randint(1, self.bb[s].spaces)            # handle of the new block bonded to the parent
                nested_lookup(chromosome, avl_pos[r]).append(self.bb[s].smiles_struct)
                for j in range(self.bb[s].spaces):
                    if (j + 1) != t:
                        nested_lookup(chromosome, avl_pos[r]).append([])
                    else:
                        nested_lookup(chromosome, avl_pos[r]).append(['C'])
            i += 1

        return deepcopy(chromosome)

    def list_to_smiles(self, indi_list):
        """
        Convert an individual (nested lists) to a molecule.

        Parameters
        ----------
        indi_list: list
            individual received from the algorithm.

        Returns
        -------
        mol_combi: str
            canonical smiles of the molecule
        """
        mol_code_list, parent_list, handle_list, mol_combi, mol_len = [], [], [], '', 0
        f_lists = count_list(indi_list)[1]

        # parent list: [list, sublist, index of sublist in list]
        parent_list.append([indi_list, indi_list, -100])
        while len(parent_list) != 0:
            iterate_over = parent_list[-1][1]
            new_item = False
            for k_ind, k in enumerate(iterate_over):
                # skip items already traversed
                if k_ind <= parent_list[-1][2]:
                    continue

                if not isinstance(k, list):
                    for mol_objs in self.bb:
                        if mol_objs.smiles_struct == k:
                            new_mol_smi = mol_objs.smiles.replace('\t', '')
                            mol_len += mol_objs.atom_len
                            break
                    mol_combi = mol_combi + new_mol_smi + '.'
                    if iterate_over == indi_list:
                        nested_ind = -50
                    else:
                        iterate_over.insert(0, 'AB')
                        for fl in f_lists:
                            if nested_lookup(indi_list, fl) == iterate_over:
                                nested_ind = fl
                                break
                        del iterate_over[0]
                    # (block code, cumulative atom count so far, nested indices of the sublist)
                    mol_code_list.append((k, mol_len, nested_ind))

                else:
                    if k:
                        if k[0] == 'C':
                            handle_2 = k_ind
                            handle_1 = parent_list[-2][2]
                            for mol_objs in self.bb:
                                if mol_objs.smiles_struct == parent_list[-1][0][0]:
                                    handle_1 = mol_objs.index_list[handle_1 - 1]
                                if mol_objs.smiles_struct == parent_list[-1][1][0]:
                                    handle_2 = mol_objs.index_list[handle_2 - 1]

                            # temporary suffixes disambiguate repeated blocks when locating their atom offsets
                            for ind, mcl in enumerate(mol_code_list):
                                if mcl[2] == -50:
                                    continue
                                if mcl[0] == parent_list[-1][0][0]:
                                    parent_list[-1][0][0] += 'WW'
                                    if 'WW' in nested_lookup(indi_list, mcl[2])[0]:
                                        handle_1 += mol_code_list[ind - 1][1]
                                    parent_list[-1][0][0] = parent_list[-1][0][0][:-2]

                                if mcl[0] == parent_list[-1][1][0]:
                                    parent_list[-1][1][0] += 'XX'
                                    if 'XX' in nested_lookup(indi_list, mcl[2])[0]:
                                        handle_2 += mol_code_list[ind - 1][1]
                                    parent_list[-1][1][0] = parent_list[-1][1][0][:-2]

                            handle_list.append([handle_1, handle_2])
                        else:
                            parent_list[-1][2] = k_ind
                            parent_list.append([iterate_over, k, -100])
                            new_item = True
                            break
            if not new_item:
                del parent_list[-1]

        mol_combi = pybel.readstring('smi', mol_combi)
        for handles in handle_list:
            mol_combi.OBMol.AddBond(handles[0], handles[1], 1)
            x = mol_combi.OBMol.GetAtom(handles[0]).GetImplicitHCount() - 1
            y = mol_combi.OBMol.GetAtom(handles[1]).GetImplicitHCount() - 1
            if x >= 0:
                mol_combi.OBMol.GetAtom(handles[0]).SetImplicitHCount(x)
            if y >= 0:
                mol_combi.OBMol.GetAtom(handles[1]).SetImplicitHCount(y)

        for atom in list(mol_combi.atoms):
            # Fr and Ra marker atoms become hydrogens
            if atom.OBAtom.GetAtomicNum() == 87 or atom.OBAtom.GetAtomicNum() == 88:
                atom.OBAtom.SetAtomicNum(1)
        return mol_combi.write("can")

    @staticmethod
    def convert_ob_to_rdkit_smiles(ob_smiles_list):
        """
        Convert Open Babel smiles to RDKit canonical smiles (explicit hydrogens removed).

        Parameters
        ----------
        ob_smiles_list: list
            smiles strings generated by Open Babel

        Returns
        -------
        conv_smis: list
        """
        from rdkit import Chem
        conv_smis = []
        for smi in ob_smiles_list:
            mol_ob = pybel.readstring("smi", smi)
            mol_ob.removeh()
            smi = mol_ob.write('can')
            smi = smi.replace('[H]', '').replace('[CH]', 'C').replace('[CH3]', 'C')
            smi = smi.split('\t')[0]
            conv_smis.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        return conv_smis

    def pre_eval(self, indi_list):
        """
        Build the molecule of an individual, check the generation rules and evaluate it.

        Returns
        -------
        mol_combi_smiles: str
        fit_val: tuple or None
            None if the molecule violates the generation rules.
        """
        mol_combi_smiles = self.list_to_smiles(deepcopy(list(indi_list)))
        mol_combi = pybel.readstring("smi", mol_combi_smiles)

        if not if_add(mol_combi_smiles, self.rules_dict, code='a'):
            return mol_combi_smiles, None

        fit_val = self.evaluate(mol_combi)
        if isinstance(fit_val, (tuple, list)):
            fit_val = tuple(fit_val)
        else:
            fit_val = (fit_val,)
        return mol_combi_smiles, fit_val

    def crossover(self, child1, child2):
        """Swap a randomly chosen filled sub-tree between two individuals."""
        child1, child2 = deepcopy(list(child1)), deepcopy(list(child2))
        c1 = count_list(child1)[1]
        c2 = count_list(child2)[1]
        if not (len(c1) == 0 or len(c2) == 0):
            r1 = self._rng.randint(0, len(c1) - 1)
            r2 = self._rng.randint(0, len(c2) - 1)
            t1 = nested_lookup(child1, c1[r1])
            t2 = nested_lookup(child2, c2[r2])
            if len(c1[r1]) > 1:
                del nested_lookup(child1, c1[r1][:-1])[c1[r1][-1]]
                nested_lookup(child1, c1[r1][:-1]).insert(c1[r1][-1], deepcopy(t2))
            else:
                del child1[c1[r1][0]]
                child1.insert(c1[r1][0], deepcopy(t2))

            if len(c2[r2]) > 1:
                del nested_lookup(child2, c2[r2][:-1])[c2[r2][-1]]
                nested_lookup(child2, c2[r2][:-1]).insert(c2[r2][-1], deepcopy(t1))
            else:
                del child2[c2[r2][0]]
                child2.insert(c2[r2][0], deepcopy(t1))
        return tuple(deepcopy(child1)), tuple(deepcopy(child2))

    def custom_mutate(self, indi):
        """Mutate an individual by adding, deleting or replacing a block (tried in random order)."""
        indi = deepcopy(list(indi))
        t = ['add', 'del', 'replace']
        self._rng.shuffle(t)
        for i in t:
            if i == 'add':
                c = count_list(indi)[0]
                if not c:
                    continue
                else:
                    r = self._rng.randint(0, len(c) - 1)
                    s = self._rng.randint(0, len(self.bb) - 1)
                    t = self._rng.randint(1, self.bb[s].spaces)
                    nested_lookup(indi, c[r]).append(self.bb[s].smiles_struct)
                    for j in range(self.bb[s].spaces):
                        if (j + 1) != t:
                            nested_lookup(indi, c[r]).append([])
                        else:
                            nested_lookup(indi, c[r]).append(['C'])
                    break
            elif i == 'del':
                c = count_list(indi)[1]
                r = self._rng.randint(0, len(c) - 1)
                if not c or len(c[r]) < 2:
                    continue
                del nested_lookup(indi, c[r])[:]
                break
            else:
                c = count_list(indi)[1]
                r = self._rng.randint(0, len(c) - 1)
                while isinstance(nested_lookup(indi, c[r])[0], list):
                    c[r].append(0)
                old_block_code = nested_lookup(indi, c[r])[0]
                for x in self.bb:
                    if x.smiles_struct == old_block_code:
                        old_block = x
                s = self._rng.randint(0, len(self.bb) - 1)
                new_block = self.bb[s]
                nested_lookup(indi, c[r])[0] = new_block.smiles_struct
                diff = old_block.spaces - new_block.spaces
                if diff < 0:
                    for _ in range(-diff):
                        nested_lookup(indi, c[r]).append([])
                elif diff > 0:
                    tmp = deepcopy(nested_lookup(indi, c[r])[1:])
                    del nested_lookup(indi, c[r])[1:]
                    nested_lookup(indi, c[r]).append(['C'])
                    for i in range(new_block.spaces - 1):
                        nested_lookup(indi, c[r]).append(self._rng.choice(tmp))
                break
        return tuple(deepcopy(indi))

    def select(self, population, fit_list, num, choice="Roulette"):
        epop, efits = [i[0] for i in fit_list], [i[1] for i in fit_list]
        o_fits = [efits[epop.index(i)] for i in population]

        df_fits = pd.DataFrame(o_fits)
        # distance from cut-offs
        weights = pd.DataFrame([df_fits[i] / self.fit_val[i][1] for i in range(df_fits.shape[1])]).T
        df_fits = df_fits * (weights.values ** [i[0] for i in self.fit_val])
        # scale to 1-2
        df2 = [((df_fits[i] - df_fits[i].min()) / (df_fits[i].max() - df_fits[i].min())) + 1 for i in range(df_fits.shape[1])]
        # invert the 'min' columns
        df2 = pd.DataFrame([df2[i] ** self.fit_val[i][0] for i in range(len(df2))]).T
        df2 = pd.DataFrame([((df2[i] - df2[i].min()) / (df2[i].max() - df2[i].min())) + 1 for i in range(df2.shape[1])])

        fitnesses = list(df2.sum())

        if choice == "Roulette":
            total_fitness = float(sum(fitnesses))
            rel_fitness = [f / total_fitness for f in fitnesses]
            probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]
            new_population = []
            for _ in range(num):
                r = self._rng.random()
                for i, individual in enumerate(population):
                    if r <= probs[i]:
                        new_population.append(deepcopy(individual))
                        break
            return new_population
        else:
            fits_sort = sorted(fitnesses, reverse=True)
            return [deepcopy(population[fitnesses.index(fits_sort[i])]) for i in range(min(num, len(population)))]

    def batch(self, fit_list):
        """
        Batch mode: write the next individuals to evaluate to ``new_molecules.csv`` (in ``output_dir``
        or the working directory). Pass an empty tuple on the first call; afterwards pass
        ``(individual, fitness tuple, smiles)`` entries for every evaluated individual.
        """
        if len(fit_list) == 0:
            pop_to_write = self.pop_generator(n=self.pop_size)
        else:
            total_pop = [i[0] for i in fit_list]
            pop = self.select(total_pop, fit_list, self.pop_size, choice="best")

            cross_pop, mutant_pop = [], []
            co_pop = self.select(pop, fit_list, self.crossover_size)
            shflist = pop + total_pop
            self._rng.shuffle(shflist)
            c_total = co_pop + shflist
            for child1, child2 in zip(c_total[::2], c_total[1::2]):
                if len(cross_pop) >= self.crossover_size:
                    break
                c1, c2 = self.crossover(child1, child2)
                epop = [i[0] for i in fit_list]
                if c1 in epop or c2 in epop or c1 == c2 or c1 in cross_pop or c2 in cross_pop:
                    continue
                cross_pop.extend([deepcopy(c1), deepcopy(c2)])

            mu_pop = self.select(pop, fit_list, self.mutation_size)
            for mutant in mu_pop + cross_pop + total_pop + pop:
                if len(mutant_pop) >= self.mutation_size:
                    break
                mt = self.custom_mutate(mutant)
                if mt in [i[0] for i in fit_list] or mt in mutant_pop:
                    continue
                mutant_pop.append(mt)
            pop_to_write = cross_pop + mutant_pop

        path = os.path.join(self.output_dir or '.', 'new_molecules.csv')
        with open(path, 'w') as fl:
            fl.write('individual,smiles\n')
            for mols in pop_to_write:
                fl.write(str(mols) + ',' + str(self.list_to_smiles(list(mols))) + '\n')

    def search(self, n_generations=20, init_ratio=0.35, crossover_ratio=0.35):
        """
        Run the search.

        Algorithm 1:
            The initial population, crossovered and mutated individuals form the pool from which the
            best n members (n = population size) become the next generation. Roulette wheel selection
            picks the parents for crossover and mutation.

        Algorithm 2:
            Same as 1, but the next generation is drawn from the pool by roulette wheel selection.

        Algorithm 3:
            Same as 1, but the best members of each of the three pools (initial, crossover, mutation)
            are selected according to ``init_ratio`` and ``crossover_ratio``.

        Algorithm 4:
            Same as 1, but the mutation population is selected from the crossover population.

        Parameters
        ----------
        n_generations: int, optional (default=20)
            number of generations

        init_ratio: float, optional (default=0.35)
            Fraction of the initial population kept for the next generation. Algorithm 3 only.

        crossover_ratio: float, optional (default=0.35)
            Fraction of the crossover population kept for the next generation. Algorithm 3 only.

        Notes
        -----
        The final population and all evaluated individuals are stored in ``self.population`` and
        ``self.fit_list`` as ``(individual, fitness tuple, smiles)``. Calling ``search`` again continues
        from there.
        """
        def fit_eval(invalid_ind, fit_list):
            epop, fit_list = [i[0] for i in fit_list], list(fit_list)
            new_pop = []
            if invalid_ind:
                invalid_ind = [i for i in invalid_ind if i not in epop]
                obval = [self.pre_eval(i) for i in invalid_ind]
                rev_smiles, fitnesses = [i[0] for i in obval], [i[1] for i in obval]
                for ind, fit, r_smi in zip(invalid_ind, fitnesses, rev_smiles):
                    if fit is not None:
                        fit_list.append((deepcopy(ind), fit, r_smi))
                        new_pop.append(deepcopy(ind))
            return tuple(fit_list), new_pop

        if init_ratio >= 1 or crossover_ratio >= 1 or (init_ratio + crossover_ratio) >= 1:
            raise ValueError("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")
        if self.population is not None:
            pop = self.population
            fit_list = self.fit_list
        else:
            pop = self.pop_generator(n=self.pop_size)
            fit_list = ()

        # fit_eval only returns newly evaluated individuals, so a resumed population must be kept as is
        fit_list, evaluated = fit_eval(pop, fit_list)
        if self.population is None:
            pop = evaluated

        generation_prefix = 'generation_'
        generation_offset = 0
        if self.output_dir and os.path.isdir(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.startswith(generation_prefix) and filename.endswith('.csv'):
                    try:
                        generation_offset = max(
                            generation_offset,
                            int(filename[len(generation_prefix):-4]))
                    except ValueError:
                        continue

        from tqdm.auto import tqdm
        pbar = tqdm(range(n_generations), desc="Generation", unit="gen", position=0, leave=True,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        total_pop = []
        for xg in pbar:
            cross_pop, mutant_pop, psum = [], [], len(fit_list)
            co_pop = self.select(pop, fit_list, self.crossover_size)
            shflist = pop + total_pop
            self._rng.shuffle(shflist)
            c_total = co_pop + shflist
            for child1, child2 in zip(c_total[::2], c_total[1::2]):
                if (len(fit_list) - psum) >= self.crossover_size:
                    break
                c1, c2 = self.crossover(child1, child2)
                epop = [i[0] for i in fit_list]
                if c1 in epop or c2 in epop or c1 == c2:
                    continue
                fit_list, new_cpop = fit_eval([c1, c2], fit_list)
                cross_pop.extend(new_cpop)

            if self.algo == 4:
                mu_pop = self.select(cross_pop, fit_list, self.mutation_size)
            else:
                mu_pop = self.select(pop, fit_list, self.mutation_size)

            pre_mu = len(fit_list)
            for mutant in mu_pop + cross_pop + total_pop + pop:
                if (len(fit_list) - pre_mu) >= self.mutation_size:
                    break
                mt = self.custom_mutate(mutant)
                if mt in [i[0] for i in fit_list]:
                    continue
                fit_list, new_mpop = fit_eval([mt], fit_list)
                mutant_pop.extend(new_mpop)

            total_pop = pop + cross_pop + mutant_pop
            if self.algo == 2:
                pop = self.select(total_pop, fit_list, self.pop_size)
            elif self.algo == 3:
                p1 = self.select(pop, fit_list, int(init_ratio * self.pop_size), choice="best")
                p2 = self.select(cross_pop, fit_list, int(crossover_ratio * self.pop_size), choice="best")
                p3 = self.select(mutant_pop, fit_list, self.pop_size - len(p1) - len(p2), choice="best")
                pop = p1 + p2 + p3
            else:
                pop = self.select(total_pop, fit_list, self.pop_size, choice="best")

            if self.output_dir is not None:
                if xg == 0:
                    df_gen = pd.DataFrame([i for i in fit_list])
                else:
                    df_gen = pd.DataFrame([i for i in fit_list[-(self.crossover_size + self.mutation_size):]])
                df_gen = df_gen[[2, 1]]
                df_gen.columns = ['Canonical SMILES', 'Fitness Values']
                os.makedirs(self.output_dir, exist_ok=True)
                generation = generation_offset + xg + 1
                df_gen.to_csv(os.path.join(self.output_dir, generation_prefix + str(generation) + '.csv'), index=False)

        pbar.close()

        self.population = pop
        self.fit_list = fit_list


def count_list(l):
    """
    Nested indices of the empty and filled lists of an individual.

    Parameters
    ----------
    l: list
        nested list

    Returns
    -------
    e_list_index: list
        nested indices of empty lists (free handles)
    f_list_index: list
        nested indices of filled lists (excluding the ['C'] bond markers)
    """
    e_list_index, f_list_index, iterate_over = [], [], []
    for e, g in zip(l, range(len(l))):
        if isinstance(e, list):
            if not e:
                temp = [g]
                e_list_index.append(temp)
            else:
                if e != ['C']:
                    temp = [g]
                    f_list_index.append(temp)
                    iterate_over.append(e)

    while len(iterate_over) != 0:
        f_list = []
        for x, prefactor in zip(iterate_over, f_list_index[-len(iterate_over):]):
            if not f_list_index:
                prefactor = []
            for e, g in zip(x, range(len(x))):
                if isinstance(e, list):
                    if e == x:                                  # self-referential lists
                        e_list_index.append(prefactor + temp)
                    else:
                        if not e:
                            temp = [g]
                            e_list_index.append(prefactor + temp)
                        else:
                            if e != ['C']:
                                temp = [g]
                                f_list_index.append(prefactor + temp)
                                f_list.append(e)

        del iterate_over[:]
        for items in f_list:
            iterate_over.append(items)
    return e_list_index, f_list_index


def nested_lookup(n, idexs):
    """
    Fetch a nested sublist given its nested indices.

    Parameters
    ----------
    n: list
        the main list
    idexs: list
        the indices of the sublist

    Returns
    -------
    list: sublist with given indices
    """
    if len(idexs) == 1:
        return n[idexs[0]]
    return nested_lookup(n[idexs[0]], idexs[1:])
