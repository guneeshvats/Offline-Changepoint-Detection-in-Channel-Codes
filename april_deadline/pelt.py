import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.base import BaseCost

############################################################################
# 1) Custom Bernoulli Cost Class for Ruptures (PELT-compatible)
############################################################################

class CostBernoulli(BaseCost):
    """
    Custom cost class for i.i.d. Bernoulli segments.
    Implements:
      - fit(signal)
      - error(start, end)
    Also defines:
      - min_size (smallest allowable segment length)
      - jump (typical step for scanning)
    so that it aligns with newer versions of ruptures.
    """

    model = "bern_custom"

    def __init__(self, min_size=1, jump=1):
        super().__init__()
        self.min_size = min_size
        self.jump = jump

    def fit(self, signal):
        """
        Keep a reference to the data and prepare precomputations (e.g. cumulative sums).
        
        Parameters
        ----------
        signal : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        self
        """
        self.signal = np.asarray(signal)
        # For 1D Bernoulli, n_features=1, but let's handle general shapes if needed.
        self.cumsums = np.cumsum(self.signal, axis=0)
        return self

    def error(self, start, end):
        """
        Negative log-likelihood of sub-signal [start, end) under Bernoulli with MLE parameter.
        """
        length = end - start
        if length <= 0:
            return 0.0
        
        # sum of 1s in [start, end)
        # careful if start=0
        if start == 0:
            seg_sum = self.cumsums[end-1]
        else:
            seg_sum = self.cumsums[end-1] - self.cumsums[start-1]
        seg_sum = np.squeeze(seg_sum)  # in 1D, make it a scalar
        
        p_hat = seg_sum / length
        
        # Handle edge cases p_hat=0 or 1 => cost=0 if data is all 0 or all 1, else inf
        if p_hat == 0.0:
            if seg_sum == 0:
                return 0.0
            else:
                return np.inf
        if p_hat == 1.0:
            if seg_sum == length:
                return 0.0
            else:
                return np.inf
        
        # General negative log-likelihood
        cost = - (seg_sum * np.log(p_hat) + (length - seg_sum)*np.log(1 - p_hat))
        return cost

    def is_fitted(self):
        """Check if fit() was called."""
        return hasattr(self, "signal")


############################################################################
# 2) Helper Functions
############################################################################

def generate_bernoulli_data(n, p, q):
    """
    Generates random data from a Bernoulli(p) => Bernoulli(q) shift at a random index.
    Returns (data, true_changepoint).
    """
    t_true = np.random.randint(1, n)
    data = np.zeros(n, dtype=int)
    data[:t_true] = np.random.rand(t_true) < p
    data[t_true:] = np.random.rand(n - t_true) < q
    return data, t_true

def detect_single_changepoint_pelt_bern(data):
    data_2d = data.reshape(-1, 1)
    cost = CostBernoulli(min_size=1, jump=1).fit(data_2d)
    algo = rpt.Pelt(custom_cost=cost).fit(data_2d)
    
    # We MUST specify a penalty. '10' is just a placeholder; you can try other values.
    bkps = algo.predict(pen=10)

    # If bkps is, e.g., [30, 70, 100], that means 2 changes: 30, 70
    # plus the last index 100. If we just want the first break:
    if len(bkps) > 1:
        t_hat = bkps[0]  # the earliest changepoint
    else:
        # if the list has length 1, that means no break found => fallback
        t_hat = 0
    return t_hat


def run_experiment_method3(
    sample_sizes=[10, 50, 100], 
    p=0.3, 
    q=0.7, 
    max_delta=10, 
    iterations=100
):
    """
    Exactly like the earlier 'run_experiment' patterns:
     - For each n in sample_sizes
     - For delta in 0..max_delta
       * Generate 100 random signals
       * Detect CP
       * Check if within delta => success
     - Store accuracy and plot all 3 curves
    """
    delta_values = range(0, max_delta+1)
    results = {}

    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0
            
            for _ in range(iterations):
                data, t_true = generate_bernoulli_data(n, p, q)
                t_hat = detect_single_changepoint_pelt_bern(data)
                if abs(t_hat - t_true) <= delta:
                    success_count += 1

            accuracy_list.append(success_count / iterations)
        
        results[n] = accuracy_list

    # Plot the accuracy curves
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    plt.title("Accuracy vs. Delta (Method 3: PELT with Bernoulli cost) - (.3, .7)")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Print out results
    print("===== Accuracy values by delta (Method 3: PELT) =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")

    return results

############################################################################
# 3) Example Usage
############################################################################

if __name__ == "__main__":
    results_method3 = run_experiment_method3(
        sample_sizes=[10, 50, 100],
        p=0.3,
        q=0.7,
        max_delta=10,
        iterations=100
    )
