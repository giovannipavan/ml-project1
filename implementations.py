"""Machine learning toolbox for the project.

Every method follows the same convention: it returns a tuple (w, loss),
where w is the last weight vector (a 1D array of shape (D,)) and loss is the
value of the cost function at that w, computed on the whole training set:

  - linear methods (mean_squared_error_gd, mean_squared_error_sgd,
    least_squares, ridge_regression): MSE with the 1/2 factor used in the
    lecture notes
        L(w) = 1 / (2N) * sum_n (y_n - x_n^T w)^2

  - logistic methods (logistic_regression, reg_logistic_regression): mean
    negative log-likelihood, with labels y in {0, 1}
        L(w) = 1 / N * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]

For the regularized methods (ridge_regression and reg_logistic_regression)
the returned loss does NOT contain the penalty term.

Notation used in the docstrings:
    N : number of samples
    D : number of features
    y : targets, shape (N,)
    tx: data matrix, shape (N, D)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers: loss and gradients
# ---------------------------------------------------------------------------


def compute_loss(y, tx, w):
    """Compute the MSE loss (with the 1/2 factor) at w.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,), the model parameters

    Returns:
        The loss, as a scalar.
    """
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.

    The gradient is -1/N * X^T (y - Xw).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        Numpy array of shape (D,), same shape as w.
    """
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient of the MSE loss at w.

    Same formula as compute_gradient, but evaluated on a small batch of B
    samples (B < N) instead of the whole dataset.

    Args:
        y: numpy array of shape (B,)
        tx: numpy array of shape (B, D)
        w: numpy array of shape (D,)

    Returns:
        Numpy array of shape (D,), same shape as w.
    """
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Iterate over mini-batches of (y, tx).

    With shuffle=True each batch starts at a random position, so batches can
    overlap. A random offset is added to the start index so that the samples
    left over when N is not a multiple of batch_size can also be drawn.
    Individual samples are never shuffled inside a batch, so for a more
    random result use batch_size=1.

    With shuffle=False the batches are taken in order, cycling back to the
    beginning when the data runs out.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        batch_size: number of samples in each batch
        num_batches: number of batches to generate
        shuffle: whether the starting positions are random

    Yields:
        Tuples (y_batch, tx_batch) with y_batch of shape (batch_size,) and
        tx_batch of shape (batch_size, D).

    Example:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            ...
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)

    # Number of non-overlapping batches, and the samples they leave out
    max_batches = data_size // batch_size
    remainder = data_size - max_batches * batch_size

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        end = start + batch_size
        yield y[start:end], tx[start:end]


# ---------------------------------------------------------------------------
# Linear regression with gradient methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (GD).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), starting point
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), the weights after the last step
        loss: MSE of w on the whole dataset
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)

    return w, compute_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (SGD).

    Each step uses a single datapoint, picked uniformly at random.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), starting point
        max_iters: number of SGD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), the weights after the last step
        loss: MSE of w on the whole dataset (not on a single sample)
    """
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        # slicing (instead of y[i]) keeps the (1,) and (1, D) shapes
        grad = compute_stoch_gradient(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad

    return w, compute_loss(y, tx, w)


def mean_squared_error_sgd_batch(y, tx, initial_w, batch_size, max_iters, gamma):
    """Linear regression using mini-batch SGD, drawing batches with batch_iter.

    Not one of the required methods: it is the batch version of
    mean_squared_error_sgd, kept to compare speed and final loss. With
    batch_size=1 it is equivalent to mean_squared_error_sgd, only slower.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), starting point
        batch_size: number of samples used for each gradient estimate
        max_iters: number of SGD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), the weights after the last step
        loss: MSE of w on the whole dataset
    """
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            w = w - gamma * compute_stoch_gradient(batch_y, batch_tx, w)

    return w, compute_loss(y, tx, w)


# ---------------------------------------------------------------------------
# Normal equations
# ---------------------------------------------------------------------------


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves (X^T X) w = X^T y. It fails with a LinAlgError if X^T X is
    singular, for example when some features are perfectly collinear.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)

    Returns:
        w: numpy array of shape (D,), the optimal weights
        loss: MSE of w on the dataset
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes 1/(2N) * ||y - Xw||^2 + lambda_ * ||w||^2. Setting the gradient
    to zero gives (X^T X + 2N * lambda_ * I) w = X^T y.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        lambda_: regularization strength, a non-negative scalar

    Returns:
        w: numpy array of shape (D,), the optimal weights
        loss: MSE of w on the dataset, WITHOUT the penalty term
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_loss(y, tx, w)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def build_poly(x, degree):
    """Polynomial basis functions for the input x, from degree 0 up to degree.

    Args:
        x: numpy array of shape (N,), a single feature
        degree: integer, highest power of the expansion

    Returns:
        Numpy array of shape (N, degree + 1), with columns x^0, x^1, ..., x^degree.
    """
    return x[:, np.newaxis] ** np.arange(degree + 1)


def build_k_indices(y, k_fold, seed):
    """Build the sample indices of each fold for k-fold cross-validation.

    The indices are shuffled with the given seed, then cut into k_fold blocks
    of equal size. If N is not a multiple of k_fold, the last N % k_fold
    shuffled indices are dropped.

    Args:
        y: numpy array of shape (N,)
        k_fold: integer, the number of folds
        seed: integer, random seed (same seed -> same folds)

    Returns:
        Numpy array of shape (k_fold, N // k_fold), one row of indices per fold.
    """
    num_row = y.shape[0]
    interval = num_row // k_fold
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    return np.array([indices[k * interval : (k + 1) * interval] for k in range(k_fold)])


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """Train ridge regression on all folds except the k-th, test on the k-th.
    Args:
        y: numpy array of shape (N,)
        x: numpy array of shape (N,), a single feature
        k_indices: numpy array returned by build_k_indices
        k: integer, the fold used as test set (not to be confused with k_fold)
        lambda_: regularization strength for ridge_regression
        degree: integer, degree of the polynomial expansion

    Returns:
        Tuple (rmse_tr, rmse_te): root mean squared errors on the training
        and on the test set, with rmse = sqrt(2 * loss).
    """
    # k-th fold as test set, all the other folds as training set
    test_idx = k_indices[k]
    train_idx = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])

    # polynomial expansion
    tx_tr = build_poly(x[train_idx], degree)
    tx_te = build_poly(x[test_idx], degree)
    y_tr, y_te = y[train_idx], y[test_idx]

    # train on the training folds only
    w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    loss_te = compute_loss(y_te, tx_te, w)

    return float(np.sqrt(2 * loss_tr)), float(np.sqrt(2 * loss_te))


def best_lambda_selection(y, x, degree, k_fold, lambdas, seed=12):
    """K-fold cross-validation over lambda, for a fixed polynomial degree.

    Args:
        y: numpy array of shape (N,)
        x: numpy array of shape (N,), a single feature
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: numpy array of shape (p,), the lambdas to test
        seed: integer, random seed used to build the folds

    Returns:
        best_lambda: scalar, the lambda with the lowest mean test RMSE
        best_rmse: scalar, the mean test RMSE for best_lambda
        rmse_tr: numpy array of shape (p,), mean train RMSE for each lambda
        rmse_te: numpy array of shape (p,), mean test RMSE for each lambda
    """
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr, rmse_te = [], []
    for lambda_ in lambdas:
        # one (train, test) RMSE pair per fold, then average over the folds
        errors = [
            cross_validation(y, x, k_indices, k, lambda_, degree) for k in range(k_fold)
        ]
        rmse_tr.append(np.mean([e[0] for e in errors]))
        rmse_te.append(np.mean([e[1] for e in errors]))

    rmse_tr = np.array(rmse_tr)
    rmse_te = np.array(rmse_te)

    idx = np.argmin(rmse_te)
    return lambdas[idx], rmse_te[idx], rmse_tr, rmse_te


def best_degree_selection(y, x, degrees, k_fold, lambdas, seed=1):
    """K-fold cross-validation over both the degree and lambda.

    Args:
        y: numpy array of shape (N,)
        x: numpy array of shape (N,), a single feature
        degrees: array of shape (d,), the degrees to test
        k_fold: integer, the number of folds
        lambdas: numpy array of shape (p,), the lambdas to test
        seed: integer, random seed used to build the folds

    Returns:
        best_degree: integer, the best degree
        best_lambda: scalar, the best lambda
        best_rmse: scalar, the mean test RMSE for (best_degree, best_lambda)
    """
    best_degree, best_lambda, best_rmse = None, None, np.inf

    for degree in degrees:
        lambda_, rmse, _, _ = best_lambda_selection(y, x, degree, k_fold, lambdas, seed)
        if rmse < best_rmse:
            best_degree, best_lambda, best_rmse = (
                int(degree),
                float(lambda_),
                float(rmse),
            )

    return best_degree, best_lambda, best_rmse


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def sigmoid(t):
    """Apply the sigmoid function, without overflow for large |t|.

    Uses exp(-|t|), which is always in (0, 1], and picks the equivalent form
    1 / (1 + e) for t >= 0 and e / (1 + e) for t < 0.

    Args:
        t: scalar or numpy array

    Returns:
        sigmoid(t), with the same shape as t
    """
    e = np.exp(-np.abs(t))
    return np.where(t >= 0, 1 / (1 + e), e / (1 + e))


def calculate_logistic_loss(y, tx, w):
    """Compute the logistic loss (mean negative log-likelihood) at w.

    Written as mean(log(1 + exp(z)) - y * z) with z = tx @ w, which is
    equivalent to the usual formula but stable for large |z|
    (np.logaddexp avoids computing exp(z) and log(0) explicitly).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        The loss, as a non-negative scalar.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss at w.

    The gradient is 1/N * X^T (sigmoid(Xw) - y).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        Numpy array of shape (D,), same shape as w.
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (GD).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), starting point
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), the weights after the last step
        loss: logistic loss of w on the whole dataset
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * calculate_logistic_gradient(y, tx, w)

    return w, calculate_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (GD).

    Minimizes the logistic loss + lambda_ * ||w||^2, so the penalty adds
    2 * lambda_ * w to the gradient.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        lambda_: regularization strength, a non-negative scalar
        initial_w: numpy array of shape (D,), starting point
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), the weights after the last step
        loss: logistic loss of w on the whole dataset, WITHOUT the penalty term
    """
    w = initial_w
    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    return w, calculate_logistic_loss(y, tx, w)
