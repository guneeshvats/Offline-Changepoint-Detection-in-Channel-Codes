import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt

# Set seed for reproducibility
np.random.seed(42)

# Parameters
T = 200  # Fixed sequence length
theta_0 = 0.3
theta_1 = 0.7
phi = 1.5
N = 5
iterations = 1000
n_values = list(range(N + 1))

# Precompute logs
log_r = np.log(theta_0 / theta_1)
log_rc = np.log((1 - theta_0) / (1 - theta_1))
log_rp = np.log(theta_1 / theta_0)
log_rcp = np.log((1 - theta_1) / (1 - theta_0))

# Generate binary sequence with changepoint
def generate_sequence(T, theta_0, theta_1):
    tau = np.random.randint(1, T)
    pre = np.random.binomial(1, theta_0, tau)
    post = np.random.binomial(1, theta_1, T - tau)
    return np.concatenate([pre, post]), tau

# MLE based changepoint detection
def detect_mle(R):
    Xt = []
    for t in range(1, len(R)):
        left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
        right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, len(R)))
        Xt.append(left + right)
    return np.argmax(Xt) + 1

# CUSUM based changepoint detection
def detect_cusum(R):
    T = len(R)
    s = np.cumsum(R - np.mean(R))
    K = np.argmax(np.abs(s))
    return K + 1

# Ruptures Binseg (parametric known θ₀, θ₁)
def detect_ruptures(R):
    algo = rpt.Binseg(model="l2").fit(R.reshape(-1, 1))
    result = algo.predict(n_bkps=1)
    return result[0]  # index of changepoint

# Compute π using Hinkley's theoretical method
def compute_theoretical_pi(N, theta_0, theta_1, phi):
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

    q = compute_q_matrix(theta_0, phi, N, N)
    q_p = compute_q_matrix(theta_1, phi, N, N)
    p = np.zeros_like(q)
    p_p = np.zeros_like(q_p)

    for l in range(N + 1):
        for m in range(N + 1):
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
                for j in range(N + 1)
                for k in range(N + 1)
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
                for j in range(N + 1)
                for k in range(N + 1)
                if j - phi * k < bound
            )
            sum_neg += term1 * term2
        pi_neg_n.append(sum_neg)

    total = sum(pi_n) + sum(pi_neg_n) + pi_0
    pi_n = [x / total for x in pi_n]
    pi_neg_n = [x / total for x in pi_neg_n]
    pi_0 /= total

    cumulative = []
    for n in range(N + 1):
        prob = pi_0
        if n > 0:
            prob += sum(pi_n[:n]) + sum(pi_neg_n[:n])
        cumulative.append(prob)
    return cumulative

# Accuracy counters
accuracy_mle = [0] * (N + 1)
accuracy_cusum = [0] * (N + 1)
accuracy_ruptures = [0] * (N + 1)

# Run experiments
for _ in range(iterations):
    R, tau = generate_sequence(T, theta_0, theta_1)
    mle_tau = detect_mle(R)
    cusum_tau = detect_cusum(R)
    ruptures_tau = detect_ruptures(R)

    for n in range(N + 1):
        if abs(mle_tau - tau) <= n:
            accuracy_mle[n] += 1
        if abs(cusum_tau - tau) <= n:
            accuracy_cusum[n] += 1
        if abs(ruptures_tau - tau) <= n:
            accuracy_ruptures[n] += 1

# Normalize
accuracy_mle = [round(x / iterations, 3) for x in accuracy_mle]
accuracy_cusum = [round(x / iterations, 3) for x in accuracy_cusum]
accuracy_ruptures = [round(x / iterations, 3) for x in accuracy_ruptures]
accuracy_pi = compute_theoretical_pi(N, theta_0, theta_1, phi)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, accuracy_mle, marker='o', label='MLE Method', color='blue')
plt.plot(n_values, accuracy_cusum, marker='s', label='CUSUM Method', color='orange')
plt.plot(n_values, accuracy_ruptures, marker='D', label='Ruptures Binseg (l2)', color='purple')
plt.plot(n_values, accuracy_pi, marker='^', label='Theoretical πₙ (Hinkley)', color='green')

# Annotate
for n in n_values:
    plt.annotate(f'{accuracy_mle[n]:.2f}', (n, accuracy_mle[n]), xytext=(0, 4), textcoords='offset points', ha='center', color='blue')
    plt.annotate(f'{accuracy_cusum[n]:.2f}', (n, accuracy_cusum[n]), xytext=(0, -10), textcoords='offset points', ha='center', color='orange')
    plt.annotate(f'{accuracy_ruptures[n]:.2f}', (n, accuracy_ruptures[n]), xytext=(0, 12), textcoords='offset points', ha='center', color='purple')
    plt.annotate(f'{accuracy_pi[n]:.2f}', (n, accuracy_pi[n]), xytext=(0, 18), textcoords='offset points', ha='center', color='green')

plt.title(f"Changepoint Detection Accuracy Comparison (T={T})")
plt.xlabel("n (Window size: |τ̂ − τ| ≤ n)")
plt.ylabel("Accuracy / Probability")
plt.legend()
plt.grid(True)
plt.xticks(n_values)
plt.tight_layout()
plt.show()
