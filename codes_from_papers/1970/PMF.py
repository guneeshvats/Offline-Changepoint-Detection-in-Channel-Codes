import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------------------
# CATEGORICAL CHANGEPOINT DETECTION USING MLE (Maximum Likelihood Estimation)
# --------------------------------------------------------------------------------------
# PMF1: Categorical distribution before the changepoint
# PMF2: Categorical distribution after the changepoint
# The goal: estimate changepoint τ using maximum likelihood (NLL minimization)
# --------------------------------------------------------------------------------------

# --------------------------
# PARAMETERS
# --------------------------
pmf1 = [0.1, 0.5, 0.4]  # Probabilities for categories [0, 1, 2] before changepoint
pmf2 = [0.6, 0.2, 0.2]  # Probabilities for categories [0, 1, 2] after changepoint
categories = [0, 1, 2]   # Set of discrete categories
max_delta = 10           # Max error tolerance to count detection as successful
iterations = 1000        # Number of random trials for each setting
sample_sizes = [10, 50, 100, 200]  # Sequence lengths to evaluate performance over

# --------------------------
# STEP 1: GENERATE SEQUENCE WITH RANDOM CHANGEPOINT
# --------------------------
def generate_categorical_data(T, pmf1, pmf2):
    """
    Generates a categorical sequence of length T with a changepoint at random index τ.
    First τ samples follow PMF1, rest follow PMF2.
    Returns:
        data: full sequence of categories [0, 1, 2]
        tau: true changepoint location (index)
    """
    tau = np.random.randint(1, T)  # Random changepoint, not at the edges
    pre = np.random.choice(categories, size=tau, p=pmf1)
    post = np.random.choice(categories, size=T - tau, p=pmf2)
    return np.concatenate([pre, post]), tau

# --------------------------
# STEP 2: NEGATIVE LOG-LIKELIHOOD FOR A SEGMENT
# --------------------------
def negative_log_likelihood_categorical(segment):
    """
    Computes negative log-likelihood for a segment using MLE.
    For a categorical variable: NLL = -sum(count_k * log(p_hat_k))
    where p_hat_k = empirical frequency of category k
    """
    length = len(segment)
    if length == 0:
        return 0.0  # Empty segment has zero cost

    # Count how many times each category appears
    counts = np.array([np.sum(segment == k) for k in categories])
    probs = counts / length  # MLE estimate: relative frequebncy

    # Handle numerical edge cases where p_hat = 0
    if np.any(probs == 0):
        if np.all(counts == 0):
            return 0.0  # All-zero segment, no data
        else:
            return np.inf  # Impossible observation under MLE

    # Main NLL Formula : using: -sum(counts * log(probs))
    nll = -np.sum(counts * np.log(probs))
    return nll

# --------------------------
# STEP 3: EXHAUSTIVE SEARCH TO FIND BEST SPLIT (τ̂)
# --------------------------
def detect_changepoint(data):
    """
    Performs exhaustive search to find changepoint τ̂ that minimizes total NLL.
    Scans all possible t in [1, T-1] and returns the one with lowest cost.
    """
    T = len(data)
    best_cost = np.inf
    best_tau = 0

    # Try every possible split point
    for t in range(1, T):
        cost = (
            negative_log_likelihood_categorical(data[:t]) +
            negative_log_likelihood_categorical(data[t:])
        )
        if cost < best_cost:
            best_cost = cost
            best_tau = t

    return best_tau

# --------------------------
# STEP 4: RUN EXPERIMENT TO EVALUATE DETECTION ACCURACY
# --------------------------
def run_experiment_categorical():
    """
    For each sequence length T and for each delta from 0 to max_delta,
    run the changepoint detection experiment multiple times and
    record the success rate (accuracy) of detection.
    """
    delta_range = list(range(max_delta + 1))
    results = {}

    for T in sample_sizes:
        acc_per_delta = []

        for delta in delta_range:
            success = 0

            for _ in range(iterations):
                data, tau_true = generate_categorical_data(T, pmf1, pmf2)
                tau_hat = detect_changepoint(data)

                # Success if estimated τ̂ is within ±delta of true τ
                if abs(tau_hat - tau_true) <= delta:
                    success += 1

            accuracy = success / iterations
            acc_per_delta.append(accuracy)

        results[T] = acc_per_delta

    return delta_range, results

# --------------------------
# RUNNING THE EXPERIMENT
# --------------------------
delta_range, results = run_experiment_categorical()

# --------------------------
# STEP 5: PLOTTING ACCURACY VS DELTA
# --------------------------
plt.figure(figsize=(10, 6))
for T in sample_sizes:
    plt.plot(delta_range, results[T], marker='o', label=f'T = {T}')
    # Label each point with accuracy value
    for d, acc in zip(delta_range, results[T]):
        plt.annotate(f'{acc:.2f}', xy=(d, acc), xytext=(0, 5), textcoords='offset points', ha='center')

plt.title("Accuracy vs Delta for Categorical Changepoint Detection (MLE Method)")
plt.xlabel("Delta (Tolerance around True Changepoint)")
plt.ylabel("Detection Accuracy")
plt.legend()
plt.grid(True)
plt.xticks(delta_range)
plt.tight_layout()
plt.show()
