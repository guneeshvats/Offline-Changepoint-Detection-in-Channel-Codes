import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

############################################################################
# 1) Data Generation
############################################################################

def generate_nodist_data(n):
    """
    Generate data with a random changepoint where the distribution changes
    from Uniform(0,1) to Uniform(2,3) at a random location t_true.
    Returns:
      data : 1D NumPy array, length n
      t_true : integer, the actual changepoint index in [1..n-1].
    """
    t_true = np.random.randint(1, n)  # pick a random CP
    data = np.empty(n)
    # first segment => Uniform(0,1)
    data[:t_true] = np.random.uniform(low=0.0, high=1.0, size=t_true)
    # second segment => Uniform(2,3)
    data[t_true:] = np.random.uniform(low=2.0, high=3.0, size=(n - t_true))
    return data, t_true

def detect_single_changepoint_kernel(data, gamma=1.0):
    """
    Detect a single changepoint in 'data' using:
      - RBF kernel cost (CostRbf) in ruptures
      - BinarySeg search
      - Request exactly 1 breakpoint
    """
    # Reshape to (n_samples, n_features). For 1D, n_features=1
    data_2d = data.reshape(-1, 1)
    
    # Create a binary-segmentation object with RBF kernel cost
    # Pass gamma=... to control kernel bandwidth
    algo = rpt.Binseg(model="rbf", params={"gamma": gamma}).fit(data_2d)
    
    # Request exactly 1 changepoint => n_bkps=1
    bkps = algo.predict(n_bkps=1)
    # bkps is a list [cp, len(data)], so the actual CP is bkps[0]
    t_hat = bkps[0]
    return t_hat

############################################################################
# 2) Experiment Loop: vary delta, measure accuracy
############################################################################

def run_experiment_kernel(
    sample_sizes=[10, 50, 100],
    max_delta=10,
    iterations=100,
    gamma=1.0
):
    """
    For each sample size in sample_sizes:
      1) for delta in [0..max_delta]
      2) run 'iterations' random trials of generate_nodist_data()
      3) detect with RBF kernel-based cost + BinSeg
      4) check if detection is within delta => success
      5) store fraction of successes => accuracy vs delta
    Plot and print results for all sample sizes in one figure.
    
    gamma : RBF kernel bandwidth. Adjust if your data scale differs.
    """
    delta_values = range(0, max_delta+1)
    results = {}

    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0

            for _ in range(iterations):
                data, t_true = generate_nodist_data(n)
                t_hat = detect_single_changepoint_kernel(data, gamma=gamma)
                if abs(t_hat - t_true) <= delta:
                    success_count += 1
            
            accuracy_list.append(success_count / iterations)
        
        results[n] = accuracy_list

    # Plot results
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    plt.title("Accuracy vs. Delta (Kernel-based, no distribution info)")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Print table of values
    print("===== Accuracy values by delta (Kernel-based) =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")

    return results

############################################################################
# 3) Example Usage
############################################################################

if __name__ == "__main__":
    results_kernel = run_experiment_kernel(
        sample_sizes=[10, 50, 100],
        max_delta=10,
        iterations=100,
        gamma=1.0  # RBF kernel bandwidth
    )
