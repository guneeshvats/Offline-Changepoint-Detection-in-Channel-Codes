import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.base import BaseCost

############################################################################
# 1) Custom Cost Class: Empirical PMF Negative Log-Likelihood
############################################################################

class CostDiscreteEmpirical(BaseCost):
    """
    A non-parametric cost for discrete data. For a segment [start, end),
    we compute an empirical PMF of that segment, then measure the negative
    log-likelihood of the segment under that PMF.

    Inherits from BaseCost for ruptures integration.
    """
    model = "discrete_np"

    def __init__(self, min_size=1, jump=1):
        super().__init__()
        self.min_size = min_size  # smallest segment size
        self.jump = jump          # typical step
        # We'll store the data in .signal after fit()

    def fit(self, signal):
        """
        Store the data as a 1D or 2D numpy array.
        For convenience, let's handle 1D signals (n_samples,) or (n_samples,1).
        """
        self.signal = np.asarray(signal)
        # If it's shape (n,) we leave it as is. If (n,1), let's flatten it:
        if self.signal.ndim == 2 and self.signal.shape[1] == 1:
            self.signal = self.signal.ravel()
        return self

    def error(self, start, end):
        """
        Compute negative log-likelihood of segment [start, end) under its own empirical PMF.
        """
        length = end - start
        if length <= 0:
            return 0.0

        # Extract sub-signal
        seg = self.signal[start:end]

        # Count frequencies of each unique value
        # np.unique returns (unique_values, counts)
        vals, counts = np.unique(seg, return_counts=True)
        # Convert to probabilities
        pmf = counts / length

        # For each point in seg, we add -log( pmf(value) )
        # Easiest is to build a dict from value -> pmf
        pmf_map = dict(zip(vals, pmf))

        # negative log-likelihood:
        # sum_{t in segment} -log( pmf_map[ seg[t] ] )
        # We'll do it directly in code:
        nll = 0.0
        for val in seg:
            p = pmf_map[val]
            # If p=0 for some reason => inf cost
            if p == 0.0:
                return np.inf
            nll -= np.log(p)

        return nll

    def is_fitted(self):
        """Check if fit() has been called."""
        return hasattr(self, "signal")


############################################################################
# 2) Generating Synthetic Discrete Data
############################################################################

def generate_discrete_data(n):
    """
    Example: data that jumps from one discrete distribution to another
    at a random location t_true.
    We'll pick a random t_true in [1..n-1].
    For demonstration, let's say:
      - segment 1: values in {0,1,2} with some random pmf
      - segment 2: values in {0,1,2,3} with a different random pmf

    This is purely to test. The cost function doesn't need to know these pmfs.
    """
    t_true = np.random.randint(1, n)
    data = np.zeros(n, dtype=int)

    # Random pmf for first segment among {0,1,2}
    pmf1 = np.random.dirichlet(alpha=[1,1,1])  # 3-cat
    # Random pmf for second segment among {0,1,2,3}
    pmf2 = np.random.dirichlet(alpha=[1,1,1,1])  # 4-cat

    # Fill first part with a random draw from pmf1
    for i in range(t_true):
        data[i] = np.random.choice([0,1,2], p=pmf1)
    # Fill second part with a random draw from pmf2
    for i in range(t_true, n):
        data[i] = np.random.choice([0,1,2,3], p=pmf2)

    return data, t_true

############################################################################
# 3) Detect Single Changepoint with BinarySeg + Our Non-parametric Cost
############################################################################

def detect_single_changepoint_np(data):
    """
    Use our custom cost (CostDiscreteEmpirical) with binary segmentation
    in ruptures. Request 1 breakpoint.
    """
    # Reshape data to (n,1) for consistency with some ruptures usage
    data_2d = data.reshape(-1, 1)

    # 1) Create our cost object
    cost = CostDiscreteEmpirical(min_size=1, jump=1).fit(data_2d)

    # 2) BinarySeg with custom cost
    algo = rpt.Binseg(custom_cost=cost).fit(data_2d)

    # 3) Predict exactly 1 breakpoint => returns [cp, len(data)]
    bkps = algo.predict(n_bkps=1)
    t_hat = bkps[0]
    return t_hat

############################################################################
# 4) Experiment: multiple trials, vary delta, measure accuracy
############################################################################

def run_experiment_cFhat_style(
    sample_sizes=[10, 50, 100],
    max_delta=10,
    iterations=100
):
    """
    Similar pattern to earlier methods:
      For each n in sample_sizes
        for delta in [0..max_delta]
          run 'iterations' random data sets
            detect CP
            if |t_hat - t_true| <= delta => success
          store success fraction
    Then plot "accuracy vs delta" for all n on same figure.
    """
    import matplotlib.pyplot as plt

    delta_values = range(0, max_delta+1)
    results = {}

    for n in sample_sizes:
        accuracy_list = []
        for delta in delta_values:
            success_count = 0
            for _ in range(iterations):
                data, t_true = generate_discrete_data(n)
                t_hat = detect_single_changepoint_np(data)
                if abs(t_hat - t_true) <= delta:
                    success_count += 1
            accuracy_list.append(success_count/iterations)
        results[n] = accuracy_list

    # Plot
    plt.figure(figsize=(8,6))
    for n in sample_sizes:
        plt.plot(delta_values, results[n], marker='o', label=f"n={n}")
    plt.title("Accuracy vs. Delta (Non-Param Empirical Cost)")
    plt.xlabel("Delta around true CP")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Print table
    print("===== Accuracy values by delta =====")
    for n in sample_sizes:
        print(f"\nSample Size n={n}:")
        for d, acc in zip(delta_values, results[n]):
            print(f"  delta={d}, accuracy={acc:.3f}")

    return results

############################################################################
# 5) Demonstration
############################################################################

if __name__ == "__main__":
    results_np = run_experiment_cFhat_style(
        sample_sizes=[10, 50, 100],
        max_delta=10,
        iterations=100
    )
