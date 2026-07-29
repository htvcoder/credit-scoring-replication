import pytest
from creditrep.models.neural.exceptions import NeuralInputError
from creditrep.models.neural.nested_cv import (
    create_early_stopping_split,
    derive_early_stopping_seed,
)


def test_split_is_deterministic_stratified_and_disjoint():
    indices = tuple(range(20))
    y = [0] * 10 + [1] * 10
    seed = derive_early_stopping_seed(
        experiment_seed=42,
        model_id="mlp_1",
        outer_fold_id="o1",
        inner_fold_id="i1",
        candidate_id=0,
    )
    first = create_early_stopping_split(indices, y, seed=seed)
    second = create_early_stopping_split(indices, y, seed=seed)
    assert first == second and not set(first.train_indices) & set(
        first.validation_indices
    )
    assert set(first.train_indices) | set(first.validation_indices) == set(indices)


def test_seed_changes_with_candidate_and_small_class_fails():
    one = derive_early_stopping_seed(
        experiment_seed=42,
        model_id="mlp_1",
        outer_fold_id="o1",
        inner_fold_id="i1",
        candidate_id=0,
    )
    two = derive_early_stopping_seed(
        experiment_seed=42,
        model_id="mlp_1",
        outer_fold_id="o1",
        inner_fold_id="i1",
        candidate_id=1,
    )
    assert one != two
    with pytest.raises(NeuralInputError):
        create_early_stopping_split((1, 2, 3), [0, 0, 1], seed=1)
