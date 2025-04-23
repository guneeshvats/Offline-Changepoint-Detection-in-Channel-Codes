import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# Parameters
# -------------------------------------------
theta_0 = 0.3  # Probability before changepoint
theta_1 = 0.7  # Probability after changepoint
phi = 1.5      # Not directly used in this code
n_values = list(range(6))  # Window sizes (0 to 5)
T_values = [10, 20, 30, 40, 50, 100, 200]  # Sequence lengths
iterations = 1000  # Number of simulations per setting

# -------------------------------------------
# Pre-computed values for log-likelihoods
# -------------------------------------------
log_r = np.log(theta_0 / theta_1)
log_rc = np.log((1 - theta_0) / (1 - theta_1))
log_rp = np.log(theta_1 / theta_0)
log_rcp = np.log((1 - theta_1) / (1 - theta_0))

# -------------------------------------------
# Sequence generation with changepoint
# -------------------------------------------
def generate_sequence(T, theta_0, theta_1):
    tau = np.random.randint(1, T)
    pre = np.random.binomial(1, theta_0, tau)
    post = np.random.binomial(1, theta_1, T - tau)
    return np.concatenate([pre, post]), tau

# -------------------------------------------
# Likelihood computation based on Hinkley
# -------------------------------------------
def compute_likelihoods(R):
    T = len(R)
    Xt = []
    for t in range(1, T):
        left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
        right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, T))
        Xt.append(left + right)
    return Xt

# -------------------------------------------
# 1. Accuracy vs Sequence Length T (fixed window size n)
# -------------------------------------------
accuracy_by_n = {n: [] for n in n_values}  # Store accuracy per n, across T

for n in n_values:
    for T in T_values:
        success_count = 0
        for _ in range(iterations):
            R, tau_true = generate_sequence(T, theta_0, theta_1)
            Xt = compute_likelihoods(R)
            tau_hat = np.argmax(Xt) + 1
            if abs(tau_hat - tau_true) <= n:
                success_count += 1
        accuracy = round(success_count / iterations, 3)
        accuracy_by_n[n].append(accuracy)

# -------------------------------------------
# 2. Accuracy vs Window Size n (fixed T)
# -------------------------------------------
accuracy_by_T = {T: [] for T in T_values}  # Store accuracy per T, across n

for T in T_values:
    for n in n_values:
        success_count = 0
        for _ in range(iterations):
            R, tau_true = generate_sequence(T, theta_0, theta_1)
            Xt = compute_likelihoods(R)
            tau_hat = np.argmax(Xt) + 1
            if abs(tau_hat - tau_true) <= n:
                success_count += 1
        accuracy = round(success_count / iterations, 3)
        accuracy_by_T[T].append(accuracy)

# -------------------------------------------
# Plot 1: Accuracy vs Sequence Length T (one line per window n)
# -------------------------------------------
plt.figure(figsize=(12, 6))
for n in n_values:
    plt.plot(T_values, accuracy_by_n[n], marker='o', label=f'n = {n}')
    for T, acc in zip(T_values, accuracy_by_n[n]):
        plt.annotate(f'{acc:.2f}', xy=(T, acc), xytext=(0, 5),
                     textcoords='offset points', ha='center')
plt.title("Accuracy vs Sequence Length T\n(Hinkley Likelihood Method)")
plt.xlabel("Sequence Length T")
plt.ylabel("Accuracy (|τ̂ − τ| ≤ n)")
plt.grid(True)
plt.legend(title="Window size n")
plt.xticks(np.arange(min(T_values), max(T_values)+1, 10))
plt.tight_layout()
plt.show()

# -------------------------------------------
# Plot 2: Accuracy vs Window Size n (one line per sequence length T)
# -------------------------------------------
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(n_values, accuracy_by_T[T], marker='s', label=f'T = {T}')
    for n, acc in zip(n_values, accuracy_by_T[T]):
        plt.annotate(f'{acc:.2f}', xy=(n, acc), xytext=(0, 5),
                     textcoords='offset points', ha='center')
plt.title("Accuracy vs Window Size n\n(Hinkley Likelihood Method)")
plt.xlabel("Window Size n (|τ̂ − τ| ≤ n)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(title="Sequence Length T")
plt.xticks(n_values)
plt.tight_layout()
plt.show()
