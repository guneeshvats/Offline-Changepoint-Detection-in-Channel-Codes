########################################################################################################################################
#                                       BERNOULLI CHANGEPOINT DETECTION IMPLEMENTATION (MLE)    
########################################################################################################################################
'''
Title: Bernoulli Changepoint Detection Implementation (MLE)

Purpose: Implements methods for detecting single changepoints in 
         Bernoulli sequences using maximum likelihood estimation

Key Features:
  - Generates synthetic Bernoulli data with single changepoint
  - Implements negative log-likelihood cost function
  - Provides changepoint detection via exhaustive search
  - Includes experimental framework for accuracy evaluation

Written by : Guneesh Vats
Dated : 14th April, 2024

'''
########################################################################################################################################
#                                                              IMPORTS    
########################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd                       

############################################################################################
#                                   GENERATE BERNOULLI DATA
############################################################################################
"""
Generates a random 1D Bernoulli data sequence of length n,
with one changepoint at a random location t_true.
Before t_true => Bernoulli(p)
After (and including) t_true => Bernoulli(q)
Returns:
    data: 1D numpy array of 0/1
    t_true: the actual changepoint index in [1..n-1]
"""
############################################################################################
############################################################################################
def generate_bernoulli_data(n, p, q):
    # Randomly choose a changepoint between 1 and n-1
    t_true = np.random.randint(1, n)  # so the CP can be anywhere from 1..n-1
    data = np.zeros(n, dtype=int)

    # Segment 1: from index 0 to t_true-1 => Bernoulli(p)
    data[:t_true] = np.random.binomial(1, p, t_true)
    # Segment 2: from index t_true to n-1 => Bernoulli(q)
    data[t_true:] = np.random.binomial(1, q, n - t_true)

    return data, t_true

############################################################################################
#                           NEGATIVE LOG LIKELIHOOD BERNOULLI
############################################################################################
"""
Computes the negative log-likelihood for a Bernoulli segment.
The MLE of p is the average of the segment,
and cost = - sum_{each sample} [ y_i log p_hat + (1 - y_i) log(1 - p_hat) ].

segment: 1D array of 0/1
Returns a float (NLL)
"""
############################################################################################
############################################################################################
def negative_log_likelihood_bernoulli(segment):

    # Avoid length-0 segment
    length = len(segment)
    if length == 0:
        return 0.0  # or could treat as no cost
    
    # MLE for Bernoulli: p_hat = average of the segment
    p_hat = np.mean(segment)
    
    # Numerically stable approach:
    #  -> log(p_hat^k * (1-p_hat)^(length-k)) = k log(p_hat) + (length-k) log(1-p_hat)
    # Here, k = sum(segment).
    # If p_hat=0 or p_hat=1, we handle edge cases:
    if p_hat == 0.0:
        # All zeros => cost = sum_{i=1..length} -log(1-0)
        # which is 0 for each 0 => actually -log(1-0)=0? Not correct. Let's do a safer approach:
        # Actually if p_hat=0 => the cost is 0 for the zero observations, but INF if we had a 1
        # So let's do it more carefully:
        if np.sum(segment) == 0:
            return 0.0
        else:
            return np.inf
    if p_hat == 1.0:
        # All ones => cost = sum_{i=1..length} -log(1^1 * 0^0?) => check if there's a single 0 => cost=inf
        if np.sum(segment) == length:
            return 0.0
        else:
            return np.inf
    
    k = np.sum(segment)
    nll = - (k * np.log(p_hat) + (length - k) * np.log(1 - p_hat))
    return nll



############################################################################################
#                          DETECT SINGLE CHANGEPOINT METHOD 
############################################################################################
"""
Title: detect_single_changepoint_method

Detect a single changepoint by scanning all possible t = 1..(n-1).
Return the t_hat that minimizes the total negative log-likelihood.
"""
############################################################################################
############################################################################################
def detect_single_changepoint_method1(data):

    n = len(data)
    best_t = 0
    best_cost = np.inf

    # We'll scan t = 1..(n-1) and compute:
    # cost = nll(data[0..t-1]) + nll(data[t..n-1])
    # The t that yields the minimal cost is the estimate.
    for t in range(1, n):
        cost_left = negative_log_likelihood_bernoulli(data[:t])
        cost_right = negative_log_likelihood_bernoulli(data[t:])
        total_cost = cost_left + cost_right
        if total_cost < best_cost:
            best_cost = total_cost
            best_t = t

    return best_t



############################################################################################
############################################################################################
"""
Title: run_experiment_method

    For each n in sample_sizes:
        1) We vary delta from 0..max_delta
        2) For each delta, we generate 'iterations' random Bernoulli data 
            (with a random changepoint each time),
            run detect_single_changepoint_method1,
            check if detection is within delta => success or not.
        3) We store average success rate = (#successes) / iterations.
    We'll plot 3 lines (for each n), accuracy vs delta.
"""
############################################################################################
############################################################################################
def run_experiment_method(
    sample_sizes=[10, 50, 100], 
    p=0.3, 
    q=0.7, 
    max_delta=10, 
    iterations=100
    ):

    delta_values = range(0, max_delta+1)
    
    # We'll store results in a dict: results[n] = [acc_delta0, acc_delta1, ...]
    results = {}
    
    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0
            
            for _ in range(iterations):
                data, t_true = generate_bernoulli_data(n, p, q)
                t_hat = detect_single_changepoint_method1(data)
                
                # check success
                if abs(t_hat - t_true) <= delta:
                    success_count += 1
            
            accuracy = success_count / iterations
            accuracy_list.append(accuracy)
        
        results[n] = accuracy_list
    
    # Now we plot
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
        # Add text annotations for each point
        for d, acc in zip(delta_values, results[n]):
            plt.annotate(f'{acc:.3f}', 
                        xy=(d, acc),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center')
    
    plt.title("Accuracy vs. Delta (Method 1, Bernoulli Changepoint (.3, .7))")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy (fraction of successful detections)")
    plt.legend()
    plt.xticks(range(max_delta + 1))  # Show all integer values from 0 to max_delta
    plt.grid(True)  # Added grid to the plot
    plt.show()
    
    # Print the values
    print("===== Accuracy values by delta =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")
    
    return results




########################################################################################################################################
#                                                              TESTING    
########################################################################################################################################

# --- Actually run the experiment (example usage) ---
if __name__ == "__main__":
    results_method1 = run_experiment_method(
        sample_sizes=[10, 30, 100],
        p=0.3,
        q=0.7,
        max_delta=10,
        iterations=1000  # Changed to 1000 iterations
    )
