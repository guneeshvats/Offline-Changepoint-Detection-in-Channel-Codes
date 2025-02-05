import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaln

##############################################################################
# Utility functions: Data generation, counting successes/failures
##############################################################################

def generate_bernoulli_data(n, p, q):
    """
    Randomly pick a change-point r in [1, n-1], generate:
      X[:r] ~ Bernoulli(p)
      X[r:] ~ Bernoulli(q)
    Returns:
      X: np.array of shape (n,)
      r: the actual change-point
    """
    r = np.random.randint(1, n)  # random change from 1..n-1
    X = np.zeros(n, dtype=int)
    X[:r] = np.random.binomial(1, p, size=r)
    X[r:] = np.random.binomial(1, q, size=n-r)
    return X, r

def count_successes_failures(X, start, end):
    """
    Return (s, f) = (# successes, # failures) in X[start:end].
    """
    segment = X[start:end]
    s = np.sum(segment)
    f = len(segment) - s
    return s, f

##############################################################################
# CASE 1: Both theta1, theta2 known
##############################################################################

def estimate_changepoint_case1(X, theta1, theta2, prior=None):
    """
    Bayesian estimate of r with known theta1,theta2 (CASE 1).
    Returns:
      r_hat, post
    """
    n = len(X)
    if prior is None:
        prior = np.ones(n) / n  # uniform prior on r=0..n-1 (or 1..n)

    log_post = np.zeros(n)
    for r in range(n):
        s1, f1 = count_successes_failures(X, 0, r)
        s2, f2 = count_successes_failures(X, r, n)

        # Log-likelihood for segment 1
        if theta1 == 0:
            ll1 = -np.inf if s1 > 0 else 0.0
        elif theta1 == 1:
            ll1 = -np.inf if f1 > 0 else 0.0
        else:
            ll1 = s1*np.log(theta1) + f1*np.log(1-theta1)

        # Log-likelihood for segment 2
        if theta2 == 0:
            ll2 = -np.inf if s2 > 0 else 0.0
        elif theta2 == 1:
            ll2 = -np.inf if f2 > 0 else 0.0
        else:
            ll2 = s2*np.log(theta2) + f2*np.log(1-theta2)

        ll = ll1 + ll2

        if prior[r] <= 0:
            log_post[r] = -np.inf
        else:
            log_post[r] = np.log(prior[r]) + ll

    max_lp = np.max(log_post)
    stable_log = log_post - max_lp
    unnorm = np.exp(stable_log)
    post = unnorm / np.sum(unnorm)

    r_hat = np.argmax(post)
    return r_hat, post

def run_simulations_example_case1(n=100, p=0.3, q=0.7, num_sims=50, delta=0):
    """
    Offline simulation for CASE 1 with delta tolerance.
    If |r_hat - r_true| <= delta, we count it as a correct detection.
    """
    estimated_changepts = []
    true_changepts = []

    for _ in range(num_sims):
        X, r_true = generate_bernoulli_data(n, p, q)
        r_hat, _ = estimate_changepoint_case1(X, p, q)
        estimated_changepts.append(r_hat)
        true_changepts.append(r_true)

    estimated_changepts = np.array(estimated_changepts)
    true_changepts = np.array(true_changepts)

    diffs = np.abs(estimated_changepts - true_changepts)
    accuracy = np.mean(diffs <= delta)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(range(num_sims), true_changepts, 'bo-', label='True CP')
    plt.plot(range(num_sims), estimated_changepts, 'rs--', label='Estimated CP')
    plt.title(f"CASE 1: Both p,q known\nn={n}, p={p}, q={q}, delta={delta}\n"+
              f"Accuracy={(accuracy*100):.1f}%")
    plt.xlabel('Simulation #')
    plt.ylabel('Change-Point Index')
    plt.legend()
    plt.grid(True)
    plt.show()

    return accuracy

##############################################################################
# CASE 2: theta1 known, theta2 unknown (Beta prior)
##############################################################################

def estimate_changepoint_case2(X, theta1,
                               alpha2=2.0, beta2=2.0,
                               prior=None):
    """
    Bayesian estimate of r, with:
      - theta1 known
      - theta2 ~ Beta(alpha2, beta2)
    We'll integrate out theta2 => Beta-function ratio for second segment.
    Returns:
      r_hat, post
    """
    n = len(X)
    if prior is None:
        prior = np.ones(n) / n

    log_post = np.zeros(n)
    from math import log

    for r in range(n):
        # First segment: known theta1
        s1, f1 = count_successes_failures(X, 0, r)
        if theta1 == 0:
            ll1 = -np.inf if s1 > 0 else 0.0
        elif theta1 == 1:
            ll1 = -np.inf if f1 > 0 else 0.0
        else:
            ll1 = s1*log(theta1) + f1*log(1-theta1)

        # Second segment: integrate over theta2 ~ Beta(alpha2,beta2)
        s2, f2 = count_successes_failures(X, r, n)
        # log of Beta( s2+alpha2, f2+beta2 ) - log Beta(alpha2, beta2)
        seg2_log = betaln(s2 + alpha2, f2 + beta2) - betaln(alpha2, beta2)

        ll = ll1 + seg2_log

        if prior[r] <= 0:
            log_post[r] = -np.inf
        else:
            log_post[r] = np.log(prior[r]) + ll

    max_lp = np.max(log_post)
    stable_log = log_post - max_lp
    unnorm = np.exp(stable_log)
    post = unnorm / np.sum(unnorm)

    r_hat = np.argmax(post)
    return r_hat, post

def run_simulations_example_case2(n=100, p=0.3, q=0.7, num_sims=50,
                                  alpha2=2.0, beta2=2.0, delta=0):
    """
    Offline simulation for CASE 2:
      - theta1 = p known
      - theta2 unknown => Beta(alpha2, beta2) prior
    If |r_hat - r_true| <= delta, we count as correct detection.
    """
    estimated_changepts = []
    true_changepts = []

    for _ in range(num_sims):
        X, r_true = generate_bernoulli_data(n, p, q)
        # We "know" theta1 = p, but we do NOT know q => we do Beta( alpha2,beta2 )
        r_hat, _ = estimate_changepoint_case2(X, theta1=p,
                                              alpha2=alpha2, beta2=beta2)
        estimated_changepts.append(r_hat)
        true_changepts.append(r_true)

    estimated_changepts = np.array(estimated_changepts)
    true_changepts = np.array(true_changepts)

    diffs = np.abs(estimated_changepts - true_changepts)
    accuracy = np.mean(diffs <= delta)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(range(num_sims), true_changepts, 'bo-', label='True CP')
    plt.plot(range(num_sims), estimated_changepts, 'rs--', label='Estimated CP')
    plt.title(f"CASE 2: theta1 known, theta2 ~ Beta({alpha2},{beta2})\n"+
              f"n={n}, p={p}, q={q}, delta={delta}\nAccuracy={(accuracy*100):.1f}%")
    plt.xlabel('Simulation #')
    plt.ylabel('Change-Point Index')
    plt.legend()
    plt.grid(True)
    plt.show()

    return accuracy

##############################################################################
# CASE 3: theta1, theta2 both unknown (independent Beta priors)
##############################################################################

def estimate_changepoint_case3(X,
                               alpha1=2.0, beta1=2.0,
                               alpha2=2.0, beta2=2.0,
                               prior=None):
    """
    Bayesian estimate of r with BOTH theta1,theta2 unknown.
    - theta1 ~ Beta(alpha1,beta1)
    - theta2 ~ Beta(alpha2,beta2), independent
    The marginal likelihood for each r is product of Beta-function ratios for
    the first segment and second segment.
    """
    n = len(X)
    if prior is None:
        prior = np.ones(n) / n

    log_post = np.zeros(n)

    for r in range(n):
        # Segment 1: [0..r]
        s1, f1 = count_successes_failures(X, 0, r)
        seg1_log = betaln(s1 + alpha1, f1 + beta1) - betaln(alpha1, beta1)

        # Segment 2: [r..n]
        s2, f2 = count_successes_failures(X, r, n)
        seg2_log = betaln(s2 + alpha2, f2 + beta2) - betaln(alpha2, beta2)

        ll = seg1_log + seg2_log

        if prior[r] <= 0:
            log_post[r] = -np.inf
        else:
            log_post[r] = np.log(prior[r]) + ll

    max_lp = np.max(log_post)
    stable_log = log_post - max_lp
    unnorm = np.exp(stable_log)
    post = unnorm / np.sum(unnorm)

    r_hat = np.argmax(post)
    return r_hat, post

def run_simulations_example_case3(n=100, p=0.3, q=0.7, num_sims=50,
                                  alpha1=2.0, beta1=2.0,
                                  alpha2=2.0, beta2=2.0,
                                  delta=0):
    """
    Offline simulation for CASE 3:
      - theta1 ~ Beta(alpha1,beta1)
      - theta2 ~ Beta(alpha2,beta2)
    We generate data using p,q but the detection pretends both unknown.
    If |r_hat - r_true| <= delta, we count as a correct detection.
    """
    estimated_changepts = []
    true_changepts = []

    for _ in range(num_sims):
        X, r_true = generate_bernoulli_data(n, p, q)
        r_hat, _ = estimate_changepoint_case3(X,
                                              alpha1=alpha1, beta1=beta1,
                                              alpha2=alpha2, beta2=beta2)
        estimated_changepts.append(r_hat)
        true_changepts.append(r_true)

    estimated_changepts = np.array(estimated_changepts)
    true_changepts = np.array(true_changepts)

    diffs = np.abs(estimated_changepts - true_changepts)
    accuracy = np.mean(diffs <= delta)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(range(num_sims), true_changepts, 'bo-', label='True CP')
    plt.plot(range(num_sims), estimated_changepts, 'rs--', label='Estimated CP')
    plt.title(f"CASE 3: Both unknown => Beta priors\n"+
              f"n={n}, p={p}, q={q}, delta={delta}\nAccuracy={(accuracy*100):.1f}%")
    plt.xlabel('Simulation #')
    plt.ylabel('Change-Point Index')
    plt.legend()
    plt.grid(True)
    plt.show()

    return accuracy

##############################################################################
# Example usage in a main() block
##############################################################################

if __name__ == "__main__":
    np.random.seed(42)

    # Example: CASE 1 with delta=2
    acc1 = run_simulations_example_case1(n=100, p=0.3, q=0.7, num_sims=50, delta=4)
    print(f"CASE 1 ACCURACY (|r_hat - r_true| <= 2) = {acc1*100:.1f}%")

    # CASE 2: We "know" theta1=p, but not theta2 => Beta(2,2)
    acc2 = run_simulations_example_case2(n=100, p=0.3, q=0.7,
                                        alpha2=2.0, beta2=2.0,
                                        num_sims=50, delta=4)
    print(f"CASE 2 ACCURACY (|r_hat - r_true| <= 2) = {acc2*100:.1f}%")

    # CASE 3: Both unknown => Beta(2,2) for each
    acc3 = run_simulations_example_case3(n=100, p=0.3, q=0.7,
                                        alpha1=2.0, beta1=2.0,
                                        alpha2=2.0, beta2=2.0,
                                        num_sims=50, delta=4)
    print(f"CASE 3 ACCURACY (|r_hat - r_true| <= 2) = {acc3*100:.1f}%")
