"""Tests for data.py: the dataset loader and the original 70/30 split."""
import numpy as np

from data import FEATURE_NAMES, TARGET_NAMES, load_data, split_data


def test_load_data_shapes():
    X, y = load_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_data_has_three_balanced_classes():
    _, y = load_data()
    classes, counts = np.unique(y, return_counts=True)
    assert classes.tolist() == [0, 1, 2]
    assert counts.tolist() == [50, 50, 50]


def test_split_sizes_are_70_30():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == len(y_train) == 105
    assert len(X_test) == len(y_test) == 45


def test_split_is_reproducible():
    X, y = load_data()
    first = split_data(X, y)
    second = split_data(X, y)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_split_keeps_every_class_in_both_sets():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    assert set(y_train.tolist()) == set(y_test.tolist()) == {0, 1, 2}


def test_target_names():
    assert TARGET_NAMES == ["setosa", "versicolor", "virginica"]


def test_feature_names_match_api_fields_in_model_order():
    assert FEATURE_NAMES == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
