import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------------
theta_0 = 0.3
theta_1 = 0.7
phi = 1.5
N = 5
T_values = [10, 30, 50, 100, 200]
iterations = 1000

# -----------------------------------------------------------------------------------
# FUNCTION: CUSUM-based Pettitt changepoint detection
# -----------------------------------------------------------------------------------
def generate_sequence(T, theta_0, theta_1):
    tau = np.random.randint(1, T)
    pre = np.random.binomial(1, theta_0, tau)
    post = np.random.binomial(1, theta_1, T - tau)
    return np.concatenate([pre, post]), tau

def pettitt_statistic(R):
    T = len(R)
    S = np.cumsum(R)
    U = np.array([t * S[-1] - T * S[t - 1] for t in range(1, T)])
    return np.argmax(np.abs(U)) + 1  # index starts at 1

# -----------------------------------------------------------------------------------
# RUN SIMULATION: Accuracy of CUSUM vs. n (window size)
# -----------------------------------------------------------------------------------
cusum_accuracy_by_T = {T: [] for T in T_values}
n_range = list(range(N + 1))

for T in T_values:
    for n in n_range:
        success = 0
        for _ in range(iterations):
            R, tau_true = generate_sequence(T, theta_0, theta_1)
            tau_hat = pettitt_statistic(R)
            if abs(tau_hat - tau_true) <= n:
                success += 1
        cusum_accuracy_by_T[T].append(success / iterations)

# -----------------------------------------------------------------------------------
# THEORETICAL π_n + π_-n + π_0 from Hinkley (Recursive)
# -----------------------------------------------------------------------------------
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

L_max, M_max = N, N
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

for n in range(1, N + 1):
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

# Normalize
total = sum(pi_n) + sum(pi_neg_n) + pi_0
pi_n = [x / total for x in pi_n]
pi_neg_n = [x / total for x in pi_neg_n]
pi_0 /= total

# Cumulative π within ±n
pi_cdf = []
for n in range(N + 1):
    prob = pi_0
    if n > 0:
        prob += sum(pi_n[:n]) + sum(pi_neg_n[:n])
    pi_cdf.append(prob)

# -----------------------------------------------------------------------------------
# COMPARISON PLOT: Theoretical π CDF vs. CUSUM empirical accuracy
# -----------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(n_range, pi_cdf, marker='o', linestyle='--', label='Theoretical CDF (π)', color='black')

# Add annotations for theoretical values
for n, val in enumerate(pi_cdf):
    plt.annotate(f'{val:.2f}', xy=(n, val), xytext=(0, 5), textcoords='offset points', ha='center')

for T in T_values:
    plt.plot(n_range, cusum_accuracy_by_T[T], marker='s', label=f'Cumulative Accuracy T={T}')
    # Add annotations for each point
    for n, val in enumerate(cusum_accuracy_by_T[T]):
        plt.annotate(f'{val:.2f}', xy=(n, val), xytext=(0, 5), textcoords='offset points', ha='center')

plt.xlabel("n (Window |τ̂ − τ| ≤ n)")
plt.ylabel("Accuracy / CDF Probability")
plt.title("CUSUM Detection Accuracy vs. Theoretical π (Hinkley)")
plt.grid(True)
plt.legend()
plt.xticks(n_range)
plt.tight_layout()
plt.show()
