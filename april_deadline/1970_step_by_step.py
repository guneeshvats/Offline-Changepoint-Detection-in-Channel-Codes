import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hinkley_changepoint_full_demo(T=30, true_tau=15, theta_0=0.3, theta_1=0.7, phi=1.5, seed=42):
    np.random.seed(seed)

    # STEP 1: Generate binary Bernoulli sequence R
    R = np.concatenate([
        np.random.binomial(1, theta_0, true_tau),
        np.random.binomial(1, theta_1, T - true_tau)
    ])
    print("Step 1: Binary sequence R (0 = failure, 1 = success):")
    print(R.tolist())

    # STEP 2: Define log-likelihood ratios
    # These are based on log P(R[i]|theta_0) / P(R[i]|theta_1), derived from Hinkley paper
    log_ratio = np.log(theta_0 / theta_1)
    log_ratio_comp = np.log((1 - theta_0) / (1 - theta_1))
    log_ratio_prime = np.log(theta_1 / theta_0)
    log_ratio_comp_prime = np.log((1 - theta_1) / (1 - theta_0))

    # STEP 3: Compute Y_i (for W) and Y'_i (for W_prime)
    # These are the step sizes in the random walks, showing evidence before/after tau
    Y = [R[i] * log_ratio + (1 - R[i]) * log_ratio_comp for i in range(true_tau)]
    Y_prime = [R[i] * log_ratio_prime + (1 - R[i]) * log_ratio_comp_prime for i in range(true_tau, T)]

    # STEP 4: Compute random walks W and W_prime using cumulative sum
    W = np.concatenate([[0], np.cumsum(Y)])
    W_prime = np.concatenate([[0], np.cumsum(Y_prime)])

    # STEP 5: Compute Xt for each candidate changepoint t
    Xt_vals = []
    for t in range(1, T):
        left = sum(R[i] * log_ratio + (1 - R[i]) * log_ratio_comp for i in range(t))
        right = sum(R[i] * log_ratio_prime + (1 - R[i]) * log_ratio_comp_prime for i in range(t, T))
        Xt_vals.append(left + right)

    estimated_tau = np.argmax(Xt_vals) + 1  # Add 1 because t starts from 1

    # STEP 6: Plot binary sequence
    plt.figure(figsize=(10, 2))
    plt.stem(range(1, T + 1), R, basefmt=" ") #, use_line_collection=True)
    plt.title("Step 1: Binary Sequence R")
    plt.xlabel("Time Step")
    plt.ylabel("R[t]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # STEP 7: Plot log-likelihood values Xt
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, T), Xt_vals, marker='s', label='Xt')
    plt.axvline(x=estimated_tau, color='red', linestyle='--', label=f'Estimated τ̂ = {estimated_tau}')
    plt.axvline(x=true_tau, color='green', linestyle='--', label=f'True τ = {true_tau}')
    plt.title("Step 5: Log-Likelihood Xt for All Candidate τ")
    plt.xlabel("Candidate τ")
    plt.ylabel("Xt Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # STEP 8: Plot Random Walks
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(W)), W, label='W (Before τ)', marker='o', color='orange')
    plt.plot(range(true_tau, T + 1), W_prime, label="W_prime (After τ)", marker='x', color='blue')
    plt.axvline(x=true_tau, color='gray', linestyle='--', label='True τ')
    plt.title("Step 4: Random Walks W and W_prime Based on Y_i and Y'_i")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Log-Likelihood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # STEP 9: Output summary
    print(f"\nStep 5: Estimated Change-Point τ̂ = {estimated_tau} (True τ = {true_tau})")
    df_likelihood = pd.DataFrame({'t': list(range(1, T)), 'Xt': Xt_vals})
    print("\nStep 5: Likelihood Table (first 10 rows):")
    print(df_likelihood.head(10).to_string(index=False))

    return {
        'R': R.tolist(),
        'estimated_tau': estimated_tau,
        'true_tau': true_tau,
        'Xt_values': df_likelihood,
        'W': W,
        'W_prime': W_prime,
        'Y': Y,
        'Y_prime': Y_prime
    }

# Run the corrected and annotated Hinkley pipeline
results_full = hinkley_changepoint_full_demo()
