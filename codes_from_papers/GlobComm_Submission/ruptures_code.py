import numpy as np
import matplotlib.pyplot as plt

# Parameters
theta_0 = 0.3
theta_1 = 0.7
n_values = list(range(6))  # Window size from 0 to 5
T_values = [10, 30, 50, 100, 200]
iterations = 1000

# Log likelihood components
log_theta0 = np.log(theta_0)
log_1_minus_theta0 = np.log(1 - theta_0)
log_theta1 = np.log(theta_1)
log_1_minus_theta1 = np.log(1 - theta_1)

# Function to generate Bernoulli sequence with one changepoint
def generate_sequence(T, theta_0, theta_1):
    tau = np.random.randint(1, T)
    pre = np.random.binomial(1, theta_0, tau)
    post = np.random.binomial(1, theta_1, T - tau)
    return np.concatenate([pre, post]), tau

# Function to compute negative log-likelihood for known theta
def compute_cost(R, t):
    left = -np.sum(R[:t] * log_theta0 + (1 - R[:t]) * log_1_minus_theta0)
    right = -np.sum(R[t:] * log_theta1 + (1 - R[t:]) * log_1_minus_theta1)
    return left + right

# Store results
accuracy_by_T = {T: [] for T in T_values}

# Run experiments
for T in T_values:
    accuracies = [0] * len(n_values)
    for _ in range(iterations):
        R, tau_true = generate_sequence(T, theta_0, theta_1)
        # Search for best changepoint
        costs = [compute_cost(R, t) for t in range(1, T)]
        tau_hat = np.argmin(costs) + 1  # +1 since t starts from 1

        error = abs(tau_hat - tau_true)
        for i, n in enumerate(n_values):
            if error <= n:
                accuracies[i] += 1
    # Normalize
    accuracy_by_T[T] = [round(acc / iterations, 3) for acc in accuracies]

# Plot results
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(n_values, accuracy_by_T[T], marker='o', label=f"T = {T}")
    for n, acc in zip(n_values, accuracy_by_T[T]):
        plt.annotate(f'{acc:.2f}', xy=(n, acc), xytext=(0, 5),
                     textcoords='offset points', ha='center')

plt.title("Accuracy vs Window Size n\nBinary Segmentation with NLL (Known θ₀, θ₁)")
plt.xlabel("Window Size n (|τ̂ − τ| ≤ n)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(title="Sequence Length T")
plt.xticks(n_values)
plt.tight_layout()
plt.show()
