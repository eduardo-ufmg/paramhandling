import numpy as np
from typing import Any


def get_nparrays(Q: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures that the primary inputs Q and y are NumPy arrays.

    This function attempts to convert the inputs to NumPy arrays without making
    a copy if they are already array-like (e.g., list of lists).

    Parameters:
        Q (Any): The similarity matrix, expected to be convertible to a 2D NumPy array.
        y (Any): The label vector, expected to be convertible to a 1D NumPy array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing Q and y as NumPy arrays.

    Raises:
        TypeError: If Q or y cannot be converted to a NumPy array.
    """
    try:
        Q_arr = np.asanyarray(Q, dtype=np.float64)
        y_arr = np.asanyarray(y)
        return Q_arr, y_arr
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Inputs Q and y must be convertible to NumPy arrays. Error: {e}"
        )


def get_classes(
    y: np.ndarray, classes: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """
    Determines the unique class labels and the total number of classes.

    If a 'classes' array is provided, it is treated as the definitive source
    of all possible class labels. Otherwise, the unique labels are inferred
    directly from the 'y' array.

    Parameters:
        y (np.ndarray): The 1D array of sample labels.
        classes (np.ndarray | None): An optional array of all unique class labels.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the array of unique labels
                                and the integer count of unique classes.
    """
    if classes is not None:
        unique_labels = np.asanyarray(classes)
        n_classes = len(unique_labels)
    else:
        unique_labels, _ = np.unique(y, return_inverse=True)
        n_classes = len(unique_labels)

    return unique_labels, n_classes


def parcheck(
    Q: Any, y: Any, factor_h: float, factor_k: float, classes: np.ndarray | None = None
) -> None:
    """
    Performs comprehensive validation of input parameters for metric functions.

    This function checks for correct types, dimensions, and value consistency
    across all provided parameters. It is designed to be a single, strict
    validation gateway for multiple metric calculation functions.

    Parameters:
        Q (Any): The similarity matrix (M, N).
        y (Any): The array of labels (M,).
        factor_h (float): The RBF kernel bandwidth scaling factor.
        factor_k (float): The nearest neighbors scaling factor.
        classes (np.ndarray | None): Optional complete list of unique class labels.

    Raises:
        TypeError: If inputs have incorrect data types (e.g., factor_h is not a float).
        ValueError: If inputs have incorrect dimensions, inconsistent shapes,
                    or values outside their expected ranges.
    """
    # 1. Check factor types and ranges
    if not isinstance(factor_h, (float, int)):
        raise TypeError(f"factor_h must be a float, but got {type(factor_h)}.")
    if not isinstance(factor_k, (float, int)):
        raise TypeError(f"factor_k must be a float, but got {type(factor_k)}.")

    if not 0.0 <= factor_h <= 1.0:
        raise ValueError(
            f"factor_h must be in the range [0.0, 1.0], but got {factor_h}."
        )
    if not 0.0 <= factor_k <= 1.0:
        raise ValueError(
            f"factor_k must be in the range [0.0, 1.0], but got {factor_k}."
        )

    # 2. Convert Q and y to NumPy arrays for consistent checking
    Q_arr, y_arr = get_nparrays(Q, y)

    # 3. Check dimensions
    if Q_arr.ndim != 2:
        raise ValueError(f"Q must be a 2D array, but has {Q_arr.ndim} dimensions.")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be a 1D array, but has {y_arr.ndim} dimensions.")

    # 4. Check shape consistency
    n_samples = Q_arr.shape[0]
    if n_samples != y_arr.shape[0]:
        raise ValueError(
            f"The number of samples in Q ({n_samples}) does not match "
            f"the number of labels in y ({y_arr.shape[0]})."
        )

    # 5. Check label data type
    if not np.issubdtype(y_arr.dtype, np.integer):
        raise TypeError(f"Labels in y must be integers, but found type {y_arr.dtype}.")

    # 6. Check class consistency
    n_features_q = Q_arr.shape[1]
    _, n_classes_y = get_classes(y_arr, classes)

    if n_features_q != n_classes_y:
        raise ValueError(
            f"The number of columns in Q ({n_features_q}) must match the "
            f"number of unique classes derived from y/classes ({n_classes_y})."
        )
