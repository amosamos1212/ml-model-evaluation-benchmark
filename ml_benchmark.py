import math
import random
import unittest


def mean(values):
    """Computes the arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence.")
    return sum(values) / len(values)


def variance(values, sample=True):
    """Computes sample (ddof=1) or population (ddof=0) variance."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute variance of an empty sequence.")
    if sample and n < 2:
        return 0.0
    m = mean(values)
    denom = n - 1 if sample else n
    return sum((x - m) ** 2 for x in values) / denom


def generate_benchmark_data(n_samples=400, noise_std=1.0, seed=2026):
    """Generates tabular features with linear and quadratic signal plus Gaussian noise."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    rng = random.Random(seed)
    true_b0 = 6.0
    true_weights = [2.2, -1.5, 0.85, 0.45]  # [x1, x2, x3, x1^2]

    X = []
    y = []
    for _ in range(n_samples):
        x1 = rng.uniform(-3.0, 3.0)
        x2 = rng.uniform(-2.0, 5.0)
        x3 = rng.gauss(0.0, 1.5)
        x1_sq = x1 ** 2
        noise = rng.gauss(0.0, noise_std)

        target = (
            true_b0
            + true_weights[0] * x1
            + true_weights[1] * x2
            + true_weights[2] * x3
            + true_weights[3] * x1_sq
            + noise
        )
        X.append([x1, x2, x3, x1_sq])
        y.append(target)

    return X, y, [true_b0] + true_weights


def fit_ridge_regression(X, y, alpha=0.05):
    """Fits Ridge regression weights [b0, b1, ..., bp] using Gaussian elimination with partial pivoting."""
    if not X or len(X) != len(y):
        raise ValueError("X and y must be non-empty and have matching lengths.")
    if alpha < 0:
        raise ValueError("Regularization parameter alpha must be non-negative.")

    p = len(X[0]) + 1  # bias term + feature count
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p

    for feats, target in zip(X, y):
        row = [1.0] + list(feats)
        for i in range(p):
            xty[i] += row[i] * target
            for j in range(p):
                xtx[i][j] += row[i] * row[j]

    # Penalize feature slopes only (index 0 is the unregularized intercept)
    for i in range(1, p):
        xtx[i][i] += alpha

    aug = [xtx[i][:] + [xty[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ArithmeticError("Design matrix is singular or ill-conditioned.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        diag = aug[col][col]
        for c in range(col, p + 1):
            aug[col][c] /= diag

        for r in range(p):
            if r != col:
                factor = aug[r][col]
                for c in range(col, p + 1):
                    aug[r][c] -= factor * aug[col][c]

    return [aug[i][p] for i in range(p)]


def fit_mean_baseline(y_train):
    """Returns the training mean as a constant baseline predictor."""
    return mean(y_train)


def predict_ridge(weights, X):
    """Generates predictions for feature matrix X given Ridge weights [bias, w1, ...]."""
    bias = weights[0]
    slopes = weights[1:]
    return [bias + sum(w * f for w, f in zip(slopes, row)) for row in X]


def compute_metrics(y_true, y_pred):
    """Computes regression evaluation metrics and residual diagnostics."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and equal in length.")

    residuals = [t - p for t, p in zip(y_true, y_pred)]
    mae = mean([abs(e) for e in residuals])
    rmse = math.sqrt(mean([e ** 2 for e in residuals]))
    y_bar = mean(y_true)
    ss_res = sum(e ** 2 for e in residuals)
    ss_tot = sum((t - y_bar) ** 2 for t in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    max_abs_err = max(abs(e) for e in residuals)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "residual_mean": round(mean(residuals), 4),
        "residual_std": round(math.sqrt(variance(residuals, sample=True)), 4),
        "max_abs_error": round(max_abs_err, 4),
    }


def cross_validate_ridge(X, y, k=5, alpha=0.05, seed=42):
    """Computes k-fold cross-validation out-of-fold RMSE mean and standard deviation."""
    if k < 2 or k > len(y):
        raise ValueError("k must be between 2 and len(y).")

    rng = random.Random(seed)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    n = len(indices)
    fold_size = n // k
    rmses = []

    for fold in range(k):
        start = fold * fold_size
        end = n if fold == k - 1 else (fold + 1) * fold_size
        val_set = set(indices[start:end])

        X_tr = [X[i] for i in range(n) if i not in val_set]
        y_tr = [y[i] for i in range(n) if i not in val_set]
        X_va = [X[i] for i in val_set]
        y_va = [y[i] for i in val_set]

        weights = fit_ridge_regression(X_tr, y_tr, alpha=alpha)
        preds = predict_ridge(weights, X_va)
        rmses.append(compute_metrics(y_va, preds)["rmse"])

    return {
        "cv_rmse_mean": round(mean(rmses), 4),
        "cv_rmse_std": round(math.sqrt(variance(rmses, sample=True)), 4),
    }


class TestBenchmarkSuite(unittest.TestCase):
    def test_parameter_recovery_low_noise(self):
        X, y, _ = generate_benchmark_data(n_samples=600, noise_std=0.05, seed=101)
        weights = fit_ridge_regression(X, y, alpha=1e-4)
        preds = predict_ridge(weights, X)
        stats = compute_metrics(y, preds)
        self.assertGreater(stats["r2"], 0.99)
        self.assertLess(stats["rmse"], 0.1)

    def test_beats_mean_baseline(self):
        X, y, _ = generate_benchmark_data(n_samples=300, noise_std=1.0, seed=7)
        split = 240
        base_val = fit_mean_baseline(y[:split])
        base_metrics = compute_metrics(y[split:], [base_val] * (len(y) - split))

        w = fit_ridge_regression(X[:split], y[:split], alpha=0.05)
        model_metrics = compute_metrics(y[split:], predict_ridge(w, X[split:]))
        self.assertLess(model_metrics["rmse"], base_metrics["rmse"])


def run_benchmark():
    X, y, ground_truth = generate_benchmark_data(n_samples=400, noise_std=1.0, seed=2026)
    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    baseline_val = fit_mean_baseline(y_train)
    baseline_test = compute_metrics(y_test, [baseline_val] * len(y_test))

    weights = fit_ridge_regression(X_train, y_train, alpha=0.05)
    test_metrics = compute_metrics(y_test, predict_ridge(weights, X_test))
    cv_summary = cross_validate_ridge(X_train, y_train, k=5, alpha=0.05, seed=42)

    print("Ground truth parameters:", [round(v, 4) for v in ground_truth])
    print("Fitted ridge weights:    ", [round(v, 4) for v in weights])
    print("Baseline test metrics:   ", baseline_test)
    print("Ridge test metrics:      ", test_metrics)
    print("5-Fold CV summary:       ", cv_summary)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBenchmarkSuite)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\n--- Running Benchmark Evaluation ---")
        run_benchmark()
