import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for θ₀, θ₁, φ (probability before/after changepoint and irrational slope)
theta_0 = 0.3
theta_1 = 0.7
phi = 1.5
N = 5  # Compute πₙ, π₋ₙ up to n = 5

# ----------------------------------------------------------------------------------------
# Step 1: Recursive computation of q_{l,m} matrix (number of sequences of failures/successes)
# ----------------------------------------------------------------------------------------
def compute_q_matrix(theta, phi, L_max, M_max):
    q = np.zeros((L_max + 1, M_max + 1))
    q[0][0] = 1.0  # Base case: no steps taken
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

# Compute q-matrices for θ₀ (before τ) and θ₁ (after τ)
L_max, M_max = N, N
q = compute_q_matrix(theta_0, phi, L_max, M_max)
q_p = compute_q_matrix(theta_1, phi, L_max, M_max)

# ----------------------------------------------------------------------------------------
# Step 2: Derive p_{l,m} and p'_{l,m} (max reached at end condition)
# ----------------------------------------------------------------------------------------
p = np.zeros_like(q)
p_p = np.zeros_like(q_p)

for l in range(L_max + 1):
    for m in range(M_max + 1):
        if -l + phi * m >= 0:
            p[l][m] = q[l][m]
        if l - phi * m >= 0:
            p_p[l][m] = q_p[l][m]

# ----------------------------------------------------------------------------------------
# Step 3: Compute π₀ = p₀₀ × p′₀₀
# ----------------------------------------------------------------------------------------
pi_0 = p[0][0] * p_p[0][0]

# ----------------------------------------------------------------------------------------
# Step 4: Compute πₙ (τ̂ = τ + n) and π₋ₙ (τ̂ = τ - n) for n = 1..N
# ----------------------------------------------------------------------------------------
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

# Normalize all π values to sum to 1
total = sum(pi_n) + sum(pi_neg_n) + pi_0
pi_n = [x / total for x in pi_n]
pi_neg_n = [x / total for x in pi_neg_n]
pi_0 /= total

# ----------------------------------------------------------------------------------------
# Step 5: Output computed π values
# ----------------------------------------------------------------------------------------
df_pi = pd.DataFrame({
    'n': list(range(1, N + 1)) + [0],
    'π_n (τ̂ < τ)': pi_n + [pi_0],
    'π_-n (τ̂ > τ)': pi_neg_n + [pi_0]
})

# ----------------------------------------------------------------------------------------
# Step 6: Plot A — Probability Mass Function of τ̂ - τ
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(df_pi['n'] - 0.1, df_pi['π_n (τ̂ < τ)'], width=0.2, label='πₙ (τ̂ < τ)', color='skyblue')
plt.bar(df_pi['n'] + 0.1, df_pi['π_-n (τ̂ > τ)'], width=0.2, label='π₋ₙ (τ̂ > τ)', color='orange')
plt.axhline(pi_0, color='gray', linestyle='--', label='π₀ (τ̂ = τ)')
plt.xticks(df_pi['n'])
plt.xlabel("n (Shift from true τ)")
plt.ylabel("Probability Mass")
plt.title("PMF of τ̂ - τ (Hinkley 1970)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------
# Step 7: Plot B — Cumulative Probability Within ±n Window
# ----------------------------------------------------------------------------------------
cumulative_probs = []
for n in range(N + 1):
    prob = pi_0
    if n > 0:
        prob += sum(pi_n[:n]) + sum(pi_neg_n[:n])
    cumulative_probs.append(prob)

plt.figure(figsize=(8, 4))
plt.plot(range(N + 1), cumulative_probs, marker='o', color='purple')
plt.title("Cumulative Probability of |τ̂ - τ| ≤ n (Hinkley 1970)")
plt.xlabel("n (Window around true τ)")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.xticks(range(N + 1))
plt.tight_layout()
plt.show()

