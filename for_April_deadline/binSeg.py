import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.base import BaseCost

############################################################################
# 1) Custom Bernoulli Cost for Ruptures
############################################################################

class CostBernoulli(BaseCost):
    """
    Custom cost class to handle Bernoulli negative log-likelihood in ruptures.
    Inherits from BaseCost and implements:
      - fit(signal)
      - error(start, end)
    """

    # A name for this model (purely for reference)
    model = "bern_custom"
    def __init__(self, min_size=2, jump=1):
        super().__init__()
        self.min_size = min_size # means no segment can be shorter than 2 samples
        self.jump = jump # means the algorithm can consider every potential index for a changepoint (instead of skipping in steps)

    def fit(self, signal):
        """
        Store the data and prepare any useful precomputations
        (e.g. partial sums) for quick error() evaluation.
        
        Parameters
        ----------
        signal: array-like, shape (n_samples, n_features)
            The data to segment.
        
        Returns
        -------
        self : object
        """
        # Ensure signal is a NumPy array
        self.signal = np.asarray(signal)
        # Typically for 1D Bernoulli, n_features=1. We'll handle the general shape carefully.

        # Precompute cumulative sums for quick sub-sum calculations.
        # We'll do it feature by feature, but in standard Bernoulli 1D use, there's just 1 dimension.
        self.cumsums = np.cumsum(self.signal, axis=0)
        
        return self

    def error(self, start, end):
        """
        Compute the negative log-likelihood of the sub-signal [start, end)
        under a Bernoulli model with MLE parameter.

        Parameters
        ----------
        start: int
            Start index (inclusive).
        end: int
            End index (exclusive).
        
        Returns
        -------
        cost : float
            The negative log-likelihood for the sub-signal [start, end).
        """
        length = end - start
        if length <= 0:
            return 0.0  # No cost if invalid segment

        # Sum of 1s in [start, end)
        # cumsums[i] = sum of signal[0..i], so sub-sum is cumsums[end-1] - cumsums[start-1].
        if start == 0:
            seg_sum = self.cumsums[end-1]
        else:
            seg_sum = self.cumsums[end-1] - self.cumsums[start-1]
        
        # If there's more than 1 feature, seg_sum might be a vector.
        # For pure 1D Bernoulli, it's shape (1,). Squeeze it:
        seg_sum = np.squeeze(seg_sum)
        
        # MLE for Bernoulli is p_hat = seg_sum / length
        p_hat = seg_sum / length

        # Edge cases: p_hat=0 or p_hat=1 => cost=0 if data is all 0 or all 1, else INF
        if p_hat == 0.0:
            if seg_sum == 0:
                return 0.0  # Perfect fit
            else:
                return np.inf
        if p_hat == 1.0:
            if seg_sum == length:
                return 0.0
            else:
                return np.inf

        # General case: negative log-likelihood
        # - [seg_sum * ln(p_hat) + (length - seg_sum)*ln(1 - p_hat)]
        cost = - (seg_sum * np.log(p_hat) + (length - seg_sum)*np.log(1 - p_hat))
        return cost

    def is_fitted(self):
        """Check if fit() has been called."""
        return hasattr(self, "signal")

############################################################################
# 2) Helper Functions for Generating Data + Running Experiments
############################################################################

def generate_bernoulli_data(n, p, q):
    """
    Same as before: create a Bernoulli( p ) => Bernoulli( q ) switch at random index.
    Returns (data, true_cp_index).
    """
    t_true = np.random.randint(1, n)  
    data = np.zeros(n, dtype=int)
    data[:t_true] = np.random.rand(t_true) < p
    data[t_true:] = np.random.rand(n - t_true) < q
    return data, t_true

def detect_single_changepoint_binseg_bern(data):
    """
    Perform Binary Segmentation using our custom Bernoulli cost in ruptures.
    We ask for exactly 1 breakpoint, so the result is the index of that breakpoint.
    """
    # Reshape data to (n_samples, n_features). For 1D, n_features=1
    data_2d = data.reshape(-1, 1)

    # 1) Create our cost object and 'fit' it (so it can do partial sums, etc.)
    cost = CostBernoulli().fit(data_2d)

    # 2) Instantiate a BinarySeg algo from ruptures, specifying the cost object
    algo = rpt.Binseg(custom_cost=cost).fit(data_2d)

    # 3) Request exactly 1 breakpoint => n_bkps=1
    bkps = algo.predict(n_bkps=1)
    # By ruptures convention, bkps is a list of the form [cp_index, len(data)].
    # If we want just the CP:
    t_hat = bkps[0]  # the actual location of the single CP
    return t_hat

def run_experiment_method2(
    sample_sizes=[10, 50, 100], 
    p=0.3, 
    q=0.7, 
    max_delta=10, 
    iterations=100
):
    """
    Same structure as method1:
      - For each n in sample_sizes
      - For delta in 0..max_delta
      - Do 'iterations' random trials
        * Generate random data + random CP
        * Detect with BinarySeg + Bernoulli cost
        * Check if detection is within delta => success
      - Store success fraction => accuracy
    Plot and print results.
    """
    delta_values = range(0, max_delta+1)
    results = {}

    for n in sample_sizes:
        accuracy_list = []
        
        for delta in delta_values:
            success_count = 0
            
            for _ in range(iterations):
                data, t_true = generate_bernoulli_data(n, p, q)
                t_hat = detect_single_changepoint_binseg_bern(data)
                
                if abs(t_hat - t_true) <= delta:
                    success_count += 1

            accuracy_list.append(success_count / iterations)
        
        results[n] = accuracy_list

    # Plot results
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    plt.title("Accuracy vs. Delta (Method 2: BinarySeg with Bernoulli cost - (.3, .7))")
    plt.xlabel("Delta (tolerance around true CP)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Print the values
    print("===== Accuracy values by delta (Method 2) =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")

    return results

############################################################################
# 3) Demonstration
############################################################################

if __name__ == "__main__":
    # Example usage
    results_method2 = run_experiment_method2(
        sample_sizes=[10, 50, 100],
        p=0.3,
        q=0.7,
        max_delta=10,
        iterations=100
    )
