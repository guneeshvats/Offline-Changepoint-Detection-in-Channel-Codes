import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# FULL HINKLEY CHANGEPOINT PIPELINE (STEP BY STEP)
# -------------------------------

# ASSUMPTIONS
# T: Length of binary sequence
# τ: True changepoint
# θ₀: Probability of 1 before τ
# θ₁: Probability of 1 after τ
# φ: Irrational constant guiding random walk comparison

# Step 1: Generate Binary Sequence R
T = 15
tau_true = 6
theta_0 = 0.3
theta_1 = 0.7
phi = 1.5
np.random.seed(42)

R = np.concatenate([
    np.random.binomial(1, theta_0, tau_true),
    np.random.binomial(1, theta_1, T - tau_true)
])
print("Step 1: Binary Sequence R:\n", R.tolist())

# Step 2: Compute Evidence Steps Yi and Yi′
log_r = np.log(theta_0 / theta_1)
log_rc = np.log((1 - theta_0) / (1 - theta_1))
log_rp = np.log(theta_1 / theta_0)
log_rcp = np.log((1 - theta_1) / (1 - theta_0))

Y = [R[i] * log_r + (1 - R[i]) * log_rc for i in range(tau_true)]
Y_prime = [R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(tau_true, T)]

print("\nStep 2: Evidence Steps Yi (before τ):", np.round(Y, 3).tolist())
print("Yi′ (after τ):", np.round(Y_prime, 3).tolist())

# Step 3: Compute and Plot Random Walks W and W′
W = np.concatenate([[0], np.cumsum(Y)])
W_prime = np.concatenate([[0], np.cumsum(Y_prime)])

plt.figure(figsize=(10, 2))
plt.stem(range(1, T + 1), R, basefmt=" ", linefmt='gray', markerfmt='D')
plt.title("Binary Sequence R")
plt.xlabel("Time")
plt.ylabel("R[t]")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 2))
plt.stem(range(1, tau_true + 1), Y, basefmt=" ", linefmt='orange', markerfmt='o', label="Yi (W)")
plt.stem(range(tau_true + 1, T + 1), Y_prime, basefmt=" ", linefmt='blue', markerfmt='x', label="Yi′ (W′)")
plt.title("Evidence Steps (Yi and Yi′)")
plt.xlabel("Time")
plt.ylabel("Log-Likelihood Increment")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(len(W)), W, marker='o', label='W (before τ)', color='orange')
plt.plot(range(tau_true, T + 1), W_prime, marker='x', label="W′ (after τ)", color='blue')
plt.axvline(tau_true, color='gray', linestyle='--', label='True τ')
plt.title("Random Walks: W and W′")
plt.xlabel("Time")
plt.ylabel("Cumulative Log-Likelihood")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Compute Xt Likelihoods for all t
Xt_vals = []
for t in range(1, T):
    xt_left = sum(R[i] * log_r + (1 - R[i]) * log_rc for i in range(t))
    xt_right = sum(R[i] * log_rp + (1 - R[i]) * log_rcp for i in range(t, T))
    Xt_vals.append(xt_left + xt_right)

estimated_tau = np.argmax(Xt_vals) + 1
print(f"\nStep 4: Estimated τ̂ = {estimated_tau} (True τ = {tau_true})")

plt.figure(figsize=(10, 3))
plt.plot(range(1, T), Xt_vals, marker='s', label='Xt')
plt.axvline(x=estimated_tau, color='red', linestyle='--', label=f'Estimated τ̂ = {estimated_tau}')
plt.axvline(x=tau_true, color='green', linestyle='--', label=f'True τ = {tau_true}')
plt.title("Likelihood Xt for Candidate τ")
plt.xlabel("Candidate τ")
plt.ylabel("Xt Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df_Xt = pd.DataFrame({'t': list(range(1, T)), 'Xt': Xt_vals})
print("\nXt Likelihood Table:")
print(df_Xt.round(3).to_string(index=False))

# Step 5: Recursive Calculation of q_{l,m} for Random Walk Distribution
def compute_q_matrix(theta, phi, L_max, M_max):
    q = np.zeros((L_max + 1, M_max + 1))
    q[0][0] = 1.0  # Default values (from Feller 1966)
    for l in range(L_max + 1):
        for m in range(M_max + 1):
            pos = -l + phi * m
            if l == 0 and m == 0:
                continue
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

# Step 6: Compute p_{l,m} = q_{l,m} where max at end
L_max, M_max = 6, 6
q = compute_q_matrix(theta_0, phi, L_max, M_max)
p = np.zeros_like(q)
for l in range(L_max + 1):
    for m in range(M_max + 1):
        if -l + phi * m >= 0:
            p[l][m] = q[l][m]

q_p = compute_q_matrix(theta_1, phi, L_max, M_max)
p_p = np.zeros_like(q_p)
for l in range(L_max + 1):
    for m in range(M_max + 1):
        if l - phi * m >= 0:
            p_p[l][m] = q_p[l][m]

# Step 7: Compute π_n and π_-n
N = 4
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

# Normalize π values (because probability)
total = sum(pi_n) + sum(pi_neg_n) + pi_0
pi_n = [x / total for x in pi_n]
pi_neg_n = [x / total for x in pi_neg_n]
pi_0 /= total

# Step 8: Output π Distribution
df_pi = pd.DataFrame({
    'n': list(range(1, N + 1)) + [0],
    'π_n': pi_n + [pi_0],
    'π_-n': pi_neg_n + [pi_0]
})
print("\nπ_n and π_-n Distribution Table:")
print(df_pi.round(4).to_string(index=False))

# Step 9: Plot π Distribution
plt.figure(figsize=(8, 4))
plt.bar(df_pi['n'] - 0.1, df_pi['π_n'], width=0.2, label='π_n (τ̂ < τ)', color='skyblue')
plt.bar(df_pi['n'] + 0.1, df_pi['π_-n'], width=0.2, label='π_-n (τ̂ > τ)', color='orange')
plt.axhline(pi_0, color='gray', linestyle='--', label='π₀ (τ̂ = τ)')
plt.xticks(df_pi['n'])
plt.xlabel("Shift n")
plt.ylabel("Probability")
plt.title("Distribution of τ̂ - τ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
