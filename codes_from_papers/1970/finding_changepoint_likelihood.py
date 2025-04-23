import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
T_values = [10, 30, 100]
theta_0 = 0.3 
theta_1 = 0.7
runs = 1000
max_window = 10

# Function to run a single trial
def run_trial(T):
    tau_true = np.random.randint(3, T - 3)
    R = np.concatenate([
        np.random.binomial(1, theta_0, tau_true),
        np.random.binomial(1, theta_1, T - tau_true)
    ])

    # Log-likelihoods
    log_r = np.log(theta_0 / theta_1)
    log_rc = np.log((1 - theta_0) / (1 - theta_1))
    log_rp = np.log(theta_1 / theta_0)
    log_rcp = np.log((1 - theta_1) / (1 - theta_0))

    # Compute Xt
    Xt_vals = []
    for t in range(1, T):
        xt_left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
        xt_right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, T))
        Xt_vals.append(xt_left + xt_right)

    estimated_tau = np.argmax(Xt_vals) + 1
    error = abs(estimated_tau - tau_true)
    return error

# Store accuracy curves for plotting
window_range = list(range(max_window + 1))
accuracy_curves = {}

for T in T_values:
    window_counts = {w: 0 for w in window_range}
    
    for _ in range(runs):
        error = run_trial(T)
        if error <= max_window:
            window_counts[error] += 1

    cumulative = 0
    cumulative_acc = []
    for w in window_range:
        cumulative += window_counts[w]
        acc = round(cumulative / runs, 3)  # Changed to probability and 3 decimal places
        cumulative_acc.append(acc)
    
    accuracy_curves[T] = cumulative_acc

    # Print accuracy table
    print(f"\n📊 Results for T = {T} (over {runs} runs):")
    for w, acc in zip(window_range, cumulative_acc):
        print(f"Within ±{w}: {acc:.3f}")

# Plot cumulative accuracy
plt.figure(figsize=(10, 6))
for T in T_values:
    plt.plot(window_range, accuracy_curves[T], marker='o', label=f"T = {T}")
    # Add text annotations for each point
    for w, acc in zip(window_range, accuracy_curves[T]):
        plt.annotate(f'{acc:.3f}', 
                    xy=(w, acc),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center')

plt.title("Cumulative Accuracy of τ̂ within Window of True τ")
plt.xlabel("Window size (|τ̂ − τ| ≤ n)")
plt.ylabel("Cumulative Accuracy (probability)")
plt.xticks(window_range)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
