import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
T = 200
theta_0 = 0.3
theta_1 = 0.7
runs = 1000

# Function to run a single simulation and return tau_hat - tau
def run_changepoint_error(T, theta_0, theta_1):
    tau_true = np.random.randint(20, T - 20)
    R = np.concatenate([
        np.random.binomial(1, theta_0, tau_true),
        np.random.binomial(1, theta_1, T - tau_true)
    ])
    
    log_r = np.log(theta_0 / theta_1)
    log_rc = np.log((1 - theta_0) / (1 - theta_1))
    log_rp = np.log(theta_1 / theta_0)
    log_rcp = np.log((1 - theta_1) / (1 - theta_0))

    Xt_vals = []
    for t in range(1, T):
        xt_left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
        xt_right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, T))
        Xt_vals.append(xt_left + xt_right)

    tau_hat = np.argmax(Xt_vals) + 1
    return tau_hat - tau_true

# Run simulations
errors = [run_changepoint_error(T, theta_0, theta_1) for _ in range(runs)]
error_counts = Counter(errors)

# Prepare data for plotting
min_error = min(error_counts)
max_error = max(error_counts)
x_vals = list(range(min_error, max_error + 1))
y_vals = [error_counts[e] / runs for e in x_vals]

# Plot empirical π-distribution
plt.figure(figsize=(10, 5))
plt.bar(x_vals, y_vals, color='purple', alpha=0.7)
plt.title("Empirical Distribution of τ̂ − τ (1000 Runs, T = 200)")
plt.xlabel("τ̂ − τ")
plt.ylabel("Probability")
plt.grid(True)
plt.tight_layout()
plt.show()
