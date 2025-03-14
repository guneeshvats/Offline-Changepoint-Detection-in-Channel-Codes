import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################################################################################
#                                   GENERATE DATA (BERNOULLI & NORMAL)
############################################################################################
"""
Generates a 1D sequence of either Bernoulli or Normal distributed data of length n,
with one changepoint at a random location t_true.

Parameters:
    n: Length of the sequence
    distribution_type: 'bernoulli' or 'normal'
    p, q: Probabilities for Bernoulli before and after changepoint
    mu1, sigma1, mu2, sigma2: Parameters for Normal distribution
Returns:
    data: 1D numpy array of values
    t_true: the actual changepoint index in [1..n-1]
"""
############################################################################################
def generate_data(n, distribution_type='bernoulli', p=0.3, q=0.7, mu1=0, sigma1=1, mu2=3, sigma2=1):
    t_true = np.random.randint(1, n)  # Random changepoint
    data = np.zeros(n)
    
    if distribution_type == 'bernoulli':
        data[:t_true] = np.random.rand(t_true) < p
        data[t_true:] = np.random.rand(n - t_true) < q
    elif distribution_type == 'normal':
        data[:t_true] = np.random.normal(mu1, sigma1, t_true)
        data[t_true:] = np.random.normal(mu2, sigma2, n - t_true)
    
    return data, t_true

############################################################################################
#                           NEGATIVE LOG LIKELIHOOD (BERNOULLI & NORMAL)
############################################################################################
def negative_log_likelihood(segment, distribution_type='bernoulli'):
    length = len(segment)
    if length == 0:
        return np.inf  # Avoid invalid cases
    
    if distribution_type == 'bernoulli':
        p_hat = np.mean(segment)
        if p_hat == 0.0 or p_hat == 1.0:
            return np.inf
        k = np.sum(segment)
        return - (k * np.log(p_hat) + (length - k) * np.log(1 - p_hat))
    
    elif distribution_type == 'normal':
        if length < 2:
            return np.inf
        mu_hat = np.mean(segment)
        sigma_hat = np.std(segment, ddof=1)
        if sigma_hat == 0:
            return np.inf
        return 0.5 * length * np.log(2 * np.pi * sigma_hat**2) + np.sum((segment - mu_hat)**2) / (2 * sigma_hat**2)

############################################################################################
#                        DETECT SINGLE CHANGEPOINT (METHOD 1)
############################################################################################
def detect_single_changepoint_method1(data, distribution_type='bernoulli'):
    n = len(data)
    best_t = 0
    best_cost = np.inf
    
    for t in range(1, n):
        cost_left = negative_log_likelihood(data[:t], distribution_type)
        cost_right = negative_log_likelihood(data[t:], distribution_type)
        total_cost = cost_left + cost_right
        if total_cost < best_cost:
            best_cost = total_cost
            best_t = t
    
    return best_t

############################################################################################
#                        RUN EXPERIMENT METHOD (BOTH DISTRIBUTIONS)
############################################################################################
def run_experiment_method(
    sample_sizes=[10, 50, 100],
    distribution_type='bernoulli',
    p=0.3, q=0.7,
    mu1=0, sigma1=1, mu2=3, sigma2=1,
    max_delta=10,
    iterations=100
    ):
    
    delta_values = range(0, max_delta+1)
    results = {}
    
    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0
            
            for _ in range(iterations):
                data, t_true = generate_data(n, distribution_type, p, q, mu1, sigma1, mu2, sigma2)
                t_hat = detect_single_changepoint_method1(data, distribution_type)
                
                if abs(t_hat - t_true) <= delta:
                    success_count += 1
            
            accuracy = success_count / iterations
            accuracy_list.append(accuracy)
        
        results[n] = accuracy_list
    
    # Now we plot
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    
    plt.title(f"Accuracy vs. Delta (Method 1, {distribution_type.capitalize()} Changepoint)")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy (fraction of successful detections)")
    plt.legend()
    plt.xticks(range(max_delta + 1))
    plt.show()
    
    print("===== Accuracy values by delta =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")
    
    return results

############################################################################################
#                                   TESTING (BOTH CASES)
############################################################################################
if __name__ == "__main__":
    print("Running Bernoulli Changepoint Detection...")
    results_bernoulli = run_experiment_method(
        sample_sizes=[10, 50, 100],
        distribution_type='bernoulli',
        p=0.3, q=0.7,
        max_delta=10,
        iterations=100
    )
    
    print("\nRunning Normal Changepoint Detection...")
    results_normal = run_experiment_method(
        sample_sizes=[10, 50, 100],
        distribution_type='normal',
        mu1=0, sigma1=1, mu2=3, sigma2=1,
        max_delta=10,
        iterations=100
    )
