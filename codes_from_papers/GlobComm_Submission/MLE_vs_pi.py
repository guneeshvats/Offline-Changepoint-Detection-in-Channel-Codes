import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Parameters and Setup
# -------------------------------
theta_0 = 0.3
theta_1 = 0.7
phi = 1.5
n_values = list(range(6))  # n = 0 to 5
T_values = [10, 30, 50, 100]
iterations = 1000

# Log-likelihood components
log_r = np.log(theta_0 / theta_1)
log_rc = np.log((1 - theta_0) / (1 - theta_1))
log_rp = np.log(theta_1 / theta_0)
log_rcp = np.log((1 - theta_1) / (1 - theta_0))

# -------------------------------
# Generate Bernoulli sequence
# -------------------------------
def generate_sequence(T, theta_0, theta_1):
    tau = np.random.randint(1, T)
    pre = np.random.binomial(1, theta_0, tau)
    post = np.random.binomial(1, theta_1, T - tau)
    return np.concatenate([pre, post]), tau

# -------------------------------
# Compute Hinkley log-likelihoods
# -------------------------------
def compute_likelihoods(R):
    T = len(R)
    Xt = []
    for t in range(1, T):
        left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
        right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, T))
        Xt.append(left + right)
    return Xt

# -------------------------------
# Step 1: MLE Accuracy vs n
# -------------------------------
accuracy_by_T = {T: [] for T in T_values}
for T in T_values:
    for n in n_values:
        success = 0
        for _ in range(iterations):
            R, tau_true = generate_sequence(T, theta_0, theta_1)
            Xt = compute_likelihoods(R)
            tau_hat = np.argmax(Xt) + 1
            if abs(tau_hat - tau_true) <= n:
                success += 1
        acc = round(success / iterations, 3)
        accuracy_by_T[T].append(acc)

# -------------------------------
# Step 2: Hinkley π₀, πₙ, π₋ₙ
# -------------------------------
def compute_q_matrix(theta, phi, L_max, M_max):
    q = np.zeros((L_max + 1, M_max + 1))
    q[0][0] = 1.0
    for l in range(L_max + 1):
        for m in range(M_max + 1):
            if l == 0 and m == 0:
                continue
            pos = -l + phi * m
            if pos < -1:
                q[l][m] = 0
            elif pos < phi:
                q[l][m] = theta * q[l - 1][m] if l > 0 else 0
            else:
                val = 0
                if l > 0:
                    val += theta * q[l - 1][m]
                if m > 0:
                    val += (1 - theta) * q[l][m - 1]
                q[l][m] = val
    return q

L_max = M_max = max(n_values)
q = compute_q_matrix(theta_0, phi, L_max, M_max)
q_p = compute_q_matrix(theta_1, phi, L_max, M_max)

p = np.zeros_like(q)
p_p = np.zeros_like(q_p)
for l in range(L_max + 1):
    for m in range(M_max + 1):
        if -l + phi * m >= 0:
            p[l][m] = q[l][m]
        if l - phi * m >= 0:
            p_p[l][m] = q_p[l][m]

pi_0 = p[0][0] * p_p[0][0]
pi_n = []
pi_neg_n = []
for n in range(1, L_max + 1):
    sum_pos = 0
    for l in range(n + 1):
        term1 = p[l][n - l]
        bound = l - phi * (n - l)
        term2 = sum(
            p_p[j][k]
            for j in range(L_max + 1)
            for k in range(M_max + 1)
            if -j + phi * k < bound
        )
        sum_pos += term1 * term2
    pi_n.append(sum_pos)

    sum_neg = 0
    for l in range(n + 1):
        term1 = p_p[l][n - l]
        bound = -l + phi * (n - l)
        term2 = sum(
            p[j][k]
            for j in range(L_max + 1)
            for k in range(M_max + 1)
            if j - phi * k < bound
        )
        sum_neg += term1 * term2
    pi_neg_n.append(sum_neg)

total = sum(pi_n) + sum(pi_neg_n) + pi_0
pi_n = [x / total for x in pi_n]
pi_neg_n = [x / total for x in pi_neg_n]
pi_0 /= total

# Compute cumulative π for |τ̂ − τ| ≤ n
cumulative_pi = []
for n in range(len(n_values)):
    prob = pi_0
    if n > 0:
        prob += sum(pi_n[:n]) + sum(pi_neg_n[:n])
    cumulative_pi.append(round(prob, 3))

# -------------------------------
# Final Plot: Accuracy vs n + π CDF
# -------------------------------
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(n_values, accuracy_by_T[T], marker='o', label=f'MLE Accuracy T={T}')
    # Add annotations for each point
    for n, val in enumerate(accuracy_by_T[T]):
        plt.annotate(f'{val:.2f}', xy=(n, val), xytext=(0, 5), textcoords='offset points', ha='center')

# Add annotations for theoretical values
for n, val in enumerate(cumulative_pi):
    plt.annotate(f'{val:.2f}', xy=(n, val), xytext=(0, 5), textcoords='offset points', ha='center')

plt.plot(n_values, cumulative_pi, linestyle='--', color='black', marker='s', label='Theoretical π CDF (Hinkley)')
plt.title("Empirical vs Theoretical Accuracy for τ̂ within Window n")
plt.xlabel("Window size n (|τ̂ − τ| ≤ n)")
plt.ylabel("Probability / Accuracy")
plt.legend()
plt.grid(True)
plt.xticks(n_values)
plt.tight_layout()
plt.show()
