import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
theta_0 = 0.3  # Probability before changepoint
theta_1 = 0.7  # Probability after changepoint
T_values = [10, 20, 30, 40, 50, 100, 200]  # Sequence lengths
n_values = list(range(6))  # Window sizes n = 0 to 5
iterations = 1000  # Number of iterations per configuration

# -------------------------------
# Step 1: Pettitt's Changepoint Estimator
# -------------------------------
def estimate_changepoint_pettitt(X):
    T = len(X)
    S = np.cumsum(X)  # Cumulative sum
    U = np.zeros(T)
    for t in range(T):
        U[t] = T * S[t] - (t + 1) * S[-1]
    tau_hat = np.argmax(U) + 1  # Estimate changepoint
    return tau_hat

# -------------------------------
# Step 2: Generate Bernoulli Sequence with Changepoint
# -------------------------------
def generate_sequence(T, theta_0, theta_1):
    tau_true = np.random.randint(3, T)
    pre = np.random.binomial(1, theta_0, tau_true)
    post = np.random.binomial(1, theta_1, T - tau_true)
    return np.concatenate([pre, post]), tau_true

# -------------------------------
# Step 3: Compute Accuracy per (T, n)
# -------------------------------
accuracy_by_T = {T: [] for T in T_values}  # Store accuracy per T across different n

for T in T_values:
    for n in n_values:
        success_count = 0
        for _ in range(iterations):
            X, tau_true = generate_sequence(T, theta_0, theta_1)
            tau_hat = estimate_changepoint_pettitt(X)
            if abs(tau_hat - tau_true) <= n:
                success_count += 1
        accuracy = round(success_count / iterations, 3)
        accuracy_by_T[T].append(accuracy)

# -------------------------------
# Step 4: Plot Accuracy vs n (Pettitt's Method)
# -------------------------------
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(n_values, accuracy_by_T[T], marker='o', label=f"T = {T}")
    for n, acc in zip(n_values, accuracy_by_T[T]):
        plt.annotate(f'{acc:.2f}', xy=(n, acc), xytext=(0, 4), textcoords='offset points', ha='center')

plt.title("Accuracy vs Window Size n (Pettitt CUSUM Method)")
plt.xlabel("Window Size n (|τ̂ − τ| ≤ n)")
plt.ylabel("Accuracy")
plt.legend(title="Sequence Length T")
plt.grid(True)
plt.xticks(n_values)
plt.tight_layout()
plt.show()
