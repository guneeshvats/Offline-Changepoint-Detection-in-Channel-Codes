import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################################################################################
#                                   GENERATE DATA (BERNOULLI, NORMAL & CATEGORICAL)
############################################################################################
def generate_data(n, distribution_type='bernoulli', p=0.3, q=0.7, mu1=0, sigma1=1, mu2=3, sigma2=1, pmf1=None, pmf2=None):
    t_true = np.random.randint(1, n)  # Random changepoint
    data = np.zeros(n, dtype=int if distribution_type == 'categorical' else float)
    
    if distribution_type == 'bernoulli':
        data[:t_true] = np.random.rand(t_true) < p
        data[t_true:] = np.random.rand(n - t_true) < q
    elif distribution_type == 'normal':
        data[:t_true] = np.random.normal(mu1, sigma1, t_true)
        data[t_true:] = np.random.normal(mu2, sigma2, n - t_true)
    elif distribution_type == 'categorical':
        data[:t_true] = np.random.choice(len(pmf1), size=t_true, p=pmf1)
        data[t_true:] = np.random.choice(len(pmf2), size=n - t_true, p=pmf2)
    
    return data, t_true

############################################################################################
#                           NEGATIVE LOG LIKELIHOOD (BERNOULLI, NORMAL & CATEGORICAL)
############################################################################################
def negative_log_likelihood(segment, distribution_type='bernoulli', num_categories=3):
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
    elif distribution_type == 'categorical':
        counts = np.bincount(segment, minlength=num_categories)
        p_hat = counts / length  # Estimate PMF
        if np.any(p_hat == 0):  # Avoid log(0) issues
            return np.inf
        return -np.sum(counts * np.log(p_hat))

############################################################################################
#                        DETECT SINGLE CHANGEPOINT (BERNOULLI, NORMAL & CATEGORICAL)
############################################################################################
def detect_single_changepoint_method1(data, distribution_type='bernoulli', num_categories=3):
    n = len(data)
    best_t = 0
    best_cost = np.inf
    
    for t in range(1, n):
        cost_left = negative_log_likelihood(data[:t], distribution_type, num_categories)
        cost_right = negative_log_likelihood(data[t:], distribution_type, num_categories)
        total_cost = cost_left + cost_right
        if total_cost < best_cost:
            best_cost = total_cost
            best_t = t
    
    return best_t

############################################################################################
#                        RUN EXPERIMENT METHOD (BERNOULLI, NORMAL & CATEGORICAL)
############################################################################################
def run_experiment_method(
    sample_sizes=[10, 50, 100],
    distribution_type='bernoulli',
    p=0.3, q=0.7,
    mu1=0, sigma1=1, mu2=3, sigma2=1,
    max_delta=10,
    iterations=100,
    num_categories=3
    ):
    
    delta_values = range(0, max_delta+1)
    results = {}
    
    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0
            
            for _ in range(iterations):
                if distribution_type == 'categorical':
                    pmf1 = np.random.dirichlet(np.ones(num_categories))
                    pmf2 = np.random.dirichlet(np.ones(num_categories))
                    data, t_true = generate_data(n, distribution_type, pmf1=pmf1, pmf2=pmf2)
                else:
                    data, t_true = generate_data(n, distribution_type, p, q, mu1, sigma1, mu2, sigma2)
                
                t_hat = detect_single_changepoint_method1(data, distribution_type, num_categories)
                
                if abs(t_hat - t_true) <= delta:
                    success_count += 1
            
            accuracy = success_count / iterations
            accuracy_list.append(accuracy)
        
        results[n] = accuracy_list
    
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    
    plt.title(f"Accuracy vs. Delta (Method 1, {distribution_type.capitalize()} Changepoint)")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy (fraction of successful detections)")
    plt.legend()
    plt.xticks(range(max_delta + 1))
    plt.show()
    
    return results

############################################################################################
#                                   TESTING (ALL CASES)
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
    
    print("\nRunning Categorical Changepoint Detection...")
    results_categorical = run_experiment_method(
        sample_sizes=[10, 50, 100],
        distribution_type='categorical',
        max_delta=10,
        iterations=100,
        num_categories=3
    )
