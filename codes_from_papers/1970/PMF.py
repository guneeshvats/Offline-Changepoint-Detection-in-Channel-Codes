# Re-import libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# PMFs and Setup
# --------------------------
pmf1 = [0.1, 0.5, 0.4]  # Before changepoint
pmf2 = [0.6, 0.2, 0.2]  # After changepoint
categories = [0, 1, 2]
max_delta = 10
iterations = 1000
sample_sizes = [10, 50, 100]

# --------------------------
# Step 1: Generate Categorical Data with Changepoint
# --------------------------
def generate_categorical_data(T, pmf1, pmf2):
    tau = np.random.randint(1, T)
    pre = np.random.choice(categories, size=tau, p=pmf1)
    post = np.random.choice(categories, size=T - tau, p=pmf2)
    return np.concatenate([pre, post]), tau

# --------------------------
# Step 2: Negative Log Likelihood for Categorical Segment
# --------------------------
def negative_log_likelihood_categorical(segment):
    length = len(segment)
    if length == 0:
        return 0.0

    counts = np.array([np.sum(segment == k) for k in categories])
    probs = counts / length

    # Handle edge case where any prob = 0
    if np.any(probs == 0):
        if np.all(counts == 0):
            return 0.0
        else:
            return np.inf

    nll = -np.sum(counts * np.log(probs))
    return nll

# --------------------------
# Step 3: Changepoint Detection via Exhaustive Search
# --------------------------
def detect_changepoint(data):
    T = len(data)
    best_cost = np.inf
    best_tau = 0
    for t in range(1, T):
        cost = negative_log_likelihood_categorical(data[:t]) + negative_log_likelihood_categorical(data[t:])
        if cost < best_cost:
            best_cost = cost
            best_tau = t
    return best_tau

# --------------------------
# Step 4: Run Experiment and Collect Accuracy
# --------------------------
def run_experiment_categorical():
    delta_range = list(range(max_delta + 1))
    results = {}

    for T in sample_sizes:
        acc_per_delta = []

        for delta in delta_range:
            success = 0

            for _ in range(iterations):
                data, tau_true = generate_categorical_data(T, pmf1, pmf2)
                tau_hat = detect_changepoint(data)
                if abs(tau_hat - tau_true) <= delta:
                    success += 1

            acc = success / iterations
            acc_per_delta.append(acc)

        results[T] = acc_per_delta

    return delta_range, results

# Run the experiment
delta_range, results = run_experiment_categorical()

# --------------------------
# Step 5: Plot Results
# --------------------------
plt.figure(figsize=(10, 6))
for T in sample_sizes:
    plt.plot(delta_range, results[T], marker='o', label=f'T = {T}')
    for d, acc in zip(delta_range, results[T]):
        plt.annotate(f'{acc:.2f}', xy=(d, acc), xytext=(0, 5), textcoords='offset points', ha='center')

plt.title("Accuracy vs Delta for Categorical Changepoint Detection")
plt.xlabel("Delta (tolerance around true changepoint)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.xticks(delta_range)
plt.tight_layout()
plt.show()
