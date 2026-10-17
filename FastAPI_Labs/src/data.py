import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

_IRIS = load_iris()

# API-friendly feature names ("sepal length (cm)" -> "sepal_length"), in the
# column order the model was trained on. These match the IrisData field names.
FEATURE_NAMES: list[str] = [
    name.replace(" (cm)", "").replace(" ", "_") for name in _IRIS.feature_names
]

# Class index -> species name, e.g. TARGET_NAMES[0] == "setosa".
TARGET_NAMES: list[str] = [str(name) for name in _IRIS.target_names]


def load_data():
    """
    Load the Iris dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Iris dataset.
        y (numpy.ndarray): The target values of the Iris dataset.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test
