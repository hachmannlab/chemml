import pytest
import multiprocessing as mp
import random
from chemml.optimization import GeneticAlgorithm

space = ({'alpha': {'uniform': [-20, 0], 
                        'mutation': [0, 2]}}, 
            {'neurons': {'int': [0,10]}},
            {'act': {'choice':range(0,100,5)}})


def evaluate(individual):
    return sum(individual)


def evaluate_raise(individual):
    if individual[1] == 5:
        raise ValueError("intentional evaluation failure")
    return sum(individual)


def _build_large_binary_space(n):
    return tuple({"f%d" % i: {'choice': [0, 1]}} for i in range(n))


def test_algorithms():
    al = [3]
    for i in al:
        ga_search = GeneticAlgorithm(
            evaluate,
            space=space,
            pop_size=10,
            mutation_size=4,
            crossover_size=4,
            algorithm=i)

        _, best_individual = ga_search.search(n_generations=4)
        assert sum([best_individual[i] for i in best_individual]) <= 200
        
def test_sequential_min():
    ga_search = GeneticAlgorithm(evaluate,
                                fitness=("min", ),
                                space=space,
                                pop_size=10,
                                mutation_size=5,
                                crossover_size=5,
                                algorithm=3)

    for _ in range(4):
        _, best_individual = ga_search.search(n_generations=1)
    assert sum([best_individual[i] for i in best_individual]) <= 200
    
def test_crossovers():
    co = ['SinglePoint', 'DoublePoint', 'Blend']
    for c in co:
        ga_search = GeneticAlgorithm(
            evaluate,
            space=space,
            crossover_type=c,
            pop_size=10,
            mutation_size=4,
            crossover_size=4,
            algorithm=3)
        _, best_individual = ga_search.search(n_generations=4)
        assert sum([best_individual[i] for i in best_individual]) <= 200


def test_parallel_evaluation_runs():
    ga_search = GeneticAlgorithm(
        evaluate,
        space=space,
        pop_size=10,
        mutation_size=4,
        crossover_size=4,
        algorithm=3,
        n_jobs=2)

    _, best_individual = ga_search.search(n_generations=2)
    assert sum([best_individual[i] for i in best_individual]) <= 200


def test_n_jobs_negative_semantics():
    ga_all = GeneticAlgorithm(evaluate, space=space, n_jobs=-1)
    ga_all_but_one = GeneticAlgorithm(evaluate, space=space, n_jobs=-2)

    tasks = 50
    expected_all = min(tasks, mp.cpu_count())
    expected_all_but_one = min(tasks, max(1, mp.cpu_count() - 1))

    assert ga_all._resolve_n_jobs(tasks) == expected_all
    assert ga_all_but_one._resolve_n_jobs(tasks) == expected_all_but_one


def test_n_jobs_zero_invalid():
    ga_search = GeneticAlgorithm(evaluate, space=space, n_jobs=0)
    with pytest.raises(ValueError, match="n_jobs=0"):
        ga_search._resolve_n_jobs(4)


def test_parallel_fallback_for_unpickleable_evaluate(capsys):
    def local_evaluate(individual):
        return sum(individual)

    ga_search = GeneticAlgorithm(
        local_evaluate,
        space=space,
        pop_size=8,
        mutation_size=3,
        crossover_size=3,
        algorithm=3,
        n_jobs=2)

    _, best_individual = ga_search.search(n_generations=1)
    assert sum([best_individual[i] for i in best_individual]) <= 200
    captured = capsys.readouterr()
    assert "Falling back to sequential evaluation" in captured.out


def test_parallel_evaluation_fail_fast_on_objective_exception():
    init_pop = [(-5.0, 5, 0), (-4.0, 3, 5), (-3.0, 2, 10), (-2.0, 1, 15)]
    ga_search = GeneticAlgorithm(
        evaluate_raise,
        space=space,
        pop_size=4,
        mutation_size=2,
        crossover_size=2,
        initial_population=init_pop,
        algorithm=3,
        n_jobs=2)

    with pytest.raises(ValueError, match="intentional evaluation failure"):
        ga_search.search(n_generations=1)


@pytest.mark.skipif(mp.get_start_method(allow_none=True) not in (None, "spawn"), reason="Windows/spawn-specific fallback behavior")
def test_parallel_fallback_for_main_module_callable(monkeypatch, capsys):
    ga_search = GeneticAlgorithm(
        evaluate,
        space=space,
        pop_size=8,
        mutation_size=3,
        crossover_size=3,
        algorithm=3,
        n_jobs=2)

    monkeypatch.setattr(ga_search.evaluate, "__module__", "__main__")
    _, best_individual = ga_search.search(n_generations=1)

    assert sum([best_individual[i] for i in best_individual]) <= 200
    captured = capsys.readouterr()
    assert "Multiprocessing disabled for evaluate defined in __main__" in captured.out


def test_large_space_feature_bias_target_features_count():
    random.seed(123)
    large_space = _build_large_binary_space(30)
    ga_search = GeneticAlgorithm(
        evaluate,
        space=large_space,
        target_features_count=6)

    assert ga_search.global_cm_list is not None
    means = [sum(ind) for ind in ga_search.global_cm_list]
    avg_selected = sum(means) / float(len(means))
    assert abs(avg_selected - 6.0) < 0.5


def test_large_space_feature_bias_active_fraction():
    random.seed(456)
    large_space = _build_large_binary_space(30)
    ga_search = GeneticAlgorithm(
        evaluate,
        space=large_space,
        active_fraction=0.1)

    assert ga_search.global_cm_list is not None
    means = [sum(ind) for ind in ga_search.global_cm_list]
    avg_selected = sum(means) / float(len(means))
    assert abs(avg_selected - 3.0) < 0.5


def test_large_space_feature_bias_strict_mutual_exclusive():
    large_space = _build_large_binary_space(30)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GeneticAlgorithm(
            evaluate,
            space=large_space,
            active_fraction=0.2,
            target_features_count=6)


def test_large_space_feature_bias_strict_scope_small_space_rejected():
    small_space = _build_large_binary_space(20)
    with pytest.raises(ValueError, match="chromosome length > 20"):
        GeneticAlgorithm(
            evaluate,
            space=small_space,
            target_features_count=5)


def test_large_space_feature_bias_strict_only_binary_choice_rejected():
    mixed_space = tuple(
        [{"f0": {'choice': [0, 1]}},
         {"f1": {'choice': [0, 1, 2]}}] +
        [{"f%d" % i: {'choice': [0, 1]}} for i in range(2, 21)]
    )
    with pytest.raises(ValueError, match="binary choice spaces"):
        GeneticAlgorithm(
            evaluate,
            space=mixed_space,
            active_fraction=0.2)


def test_large_space_feature_bias_strict_value_checks():
    large_space = _build_large_binary_space(30)

    with pytest.raises(ValueError, match=r"active_fraction must be in \[0, 1\]"):
        GeneticAlgorithm(evaluate, space=large_space, active_fraction=1.1)

    with pytest.raises(ValueError, match="target_features_count must be >= 0"):
        GeneticAlgorithm(evaluate, space=large_space, target_features_count=-1)

    with pytest.raises(ValueError, match="cannot be greater than the number of binary genes"):
        GeneticAlgorithm(evaluate, space=large_space, target_features_count=31)

