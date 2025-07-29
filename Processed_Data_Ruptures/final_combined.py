import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bernoulli
import ruptures as rpt

# =============================================================================
# Common Utility Functions for Sequence Generation and Plotting
# =============================================================================

def generate_sequence(N, changepoint, q1, q2):
    """
    Generate a binary sequence of length N with a single changepoint.
    The first 'changepoint' samples are drawn from Bernoulli(q1),
    and the remaining samples are drawn from Bernoulli(q2).

    Parameters:
        N (int): Total length of the sequence.
        changepoint (int): Index at which the changepoint occurs (0-based).
        q1 (float): Probability of 1 before the changepoint.
        q2 (float): Probability of 1 after the changepoint.

    Returns:
        np.ndarray: Binary sequence of length N.
    """
    # Generate samples before and after the changepoint using Bernoulli distributions
    before_cp = bernoulli.rvs(q1, size=changepoint)
    after_cp = bernoulli.rvs(q2, size=N - changepoint)
    return np.concatenate([before_cp, after_cp])

def smart_buffer(N):
    """
    Compute a buffer size to avoid placing changepoints too close to the sequence edges.
    For short sequences, use N//5 (minimum 1). For longer sequences, use a fixed buffer of 10.

    Parameters:
        N (int): Sequence length.

    Returns:
        int: Buffer size to use for changepoint placement.
    """
    return max(1, N // 5) if N <= 30 else 10

def plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2, title):
    """
    Plot detection accuracy versus tolerance window for multiple sequence lengths.
    Also prints a table of accuracy values for each tolerance and sequence size.

    Parameters:
        all_accuracies (list of lists): Each sublist contains accuracy per tolerance for a sequence length.
        max_tolerance (int): Maximum tolerance window.
        seq_lengths (list): List of sequence lengths.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        title (str): Title for the plot.
    """
    # Prepare and print a table of accuracy values for each tolerance and sequence length
    table_data = []
    for acc_list, N in zip(all_accuracies, seq_lengths):
        for tol, acc in enumerate(acc_list):
            table_data.append({"Seq Length": N, "Tolerance": tol, "Accuracy": acc})
    df = pd.DataFrame(table_data)
    pivot = df.pivot(index="Tolerance", columns="Seq Length", values="Accuracy")
    print("\nAccuracy Table (rows: Tolerance, columns: Sequence Length):")
    print(pivot.to_string(float_format="{:.4f}".format))
    print("-" * 60)

    # Plot accuracy curves for each sequence length
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(seq_lengths)))
    for acc_list, N, c in zip(all_accuracies, seq_lengths, colors):
        plt.plot(
            range(max_tolerance + 1),
            acc_list,
            marker='s',
            label=f"Length={N}",
            color=c
        )
        # Annotate each point with its accuracy value
        for x, y in zip(range(max_tolerance + 1), acc_list):
            plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color=c)
    plt.title(f"{title}\nq1={q1:.3f}, q2={q2:.3f}")
    plt.xlabel("Tolerance Window")
    plt.ylabel("Detection Accuracy")
    plt.xticks(np.arange(0, max_tolerance + 1, 0.5))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# Method 1: Maximum Likelihood Estimation (MLE) for Changepoint Detection
# =============================================================================

def log_likelihood(seq, tau, q1, q2):
    """
    Compute the log-likelihood of a changepoint at position tau for a binary sequence.
    The sequence is assumed to be Bernoulli(q1) before tau and Bernoulli(q2) after tau.

    Parameters:
        seq (np.ndarray): Binary sequence.
        tau (int): Changepoint candidate index.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.

    Returns:
        float: Log-likelihood value for the given changepoint.
    """
    # Calculate log-likelihood for the two segments
    ll_before = np.sum(seq[:tau] * np.log(q1) + (1 - seq[:tau]) * np.log(1 - q1))
    ll_after = np.sum(seq[tau:] * np.log(q2) + (1 - seq[tau:]) * np.log(1 - q2))
    return ll_before + ll_after

def detect_mle(seq, q1, q2, min_cp, max_cp):
    """
    Detect changepoint using MLE by maximizing the log-likelihood over all possible changepoints.

    Parameters:
        seq (np.ndarray): Binary sequence.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        min_cp (int): Minimum changepoint index to consider.
        max_cp (int): Maximum changepoint index to consider.

    Returns:
        int: Detected changepoint index (the tau with highest likelihood).
    """
    best_tau, max_ll = None, -np.inf
    for tau in range(min_cp, max_cp + 1):
        ll = log_likelihood(seq, tau, q1, q2)
        if ll > max_ll:
            best_tau = tau
            max_ll = ll
    return best_tau

def run_simulation_mle(N, q1, q2, num_iterations, max_tolerance):
    """
    Run multiple simulations of MLE changepoint detection and compute accuracy for each tolerance window.

    Parameters:
        N (int): Sequence length.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        num_iterations (int): Number of simulation runs.
        max_tolerance (int): Maximum tolerance window.

    Returns:
        list: Accuracy per tolerance window (list of floats).
    """
    buffer = smart_buffer(N)
    min_cp, max_cp = buffer, N - buffer
    if max_cp <= min_cp:
        # Not enough room for a changepoint
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    for tol in range(max_tolerance + 1):
        correct = 0
        for _ in range(num_iterations):
            cp = np.random.randint(min_cp, max_cp)
            seq = generate_sequence(N, cp, q1, q2)
            detected = detect_mle(seq, q1, q2, min_cp, max_cp)
            if abs(detected - cp) <= tol:
                correct += 1
        accuracy_per_tol.append(correct / num_iterations)
    return accuracy_per_tol

# =============================================================================
# Method 2: Ruptures Binary Segmentation (BinSeg) with Custom Bernoulli Cost
# =============================================================================

class CostBernoulli:
    """
    Custom cost function for Bernoulli-distributed binary sequences.
    This class implements the negative log-likelihood cost for a segment,
    which is used by the ruptures package for changepoint detection.
    """
    def fit(self, signal):
        """
        Fit the cost function to the input signal (binary sequence).

        Parameters:
            signal (array-like): Input binary sequence.

        Returns:
            self: The fitted cost function object.
        """
        self.signal = np.array(signal).astype(int)
        return self

    def cost(self, start, end):
        """
        Compute the negative log-likelihood cost for a segment of the sequence.

        Parameters:
            start (int): Start index of the segment (inclusive).
            end (int): End index of the segment (exclusive).

        Returns:
            float: Negative log-likelihood cost for the segment.
        """
        segment = self.signal[start:end]
        if len(segment) == 0:
            return 0
        p_hat = np.mean(segment)
        if p_hat in [0, 1]:
            # If the segment is all 0s or all 1s, log-likelihood is zero (no uncertainty)
            return 0
        n1 = np.sum(segment)
        n0 = len(segment) - n1
        return -(n1 * np.log(p_hat) + n0 * np.log(1 - p_hat))

def detect_ruptures_binseg(seq):
    """
    Detect changepoint using ruptures' Binary Segmentation algorithm with a custom Bernoulli cost.

    Parameters:
        seq (np.ndarray): Binary sequence.

    Returns:
        int or None: Detected changepoint index, or None if not found.
    """
    model = CostBernoulli().fit(seq)
    algo = rpt.Binseg(custom_cost=model).fit(seq)
    result = algo.predict(n_bkps=1)  # Only one changepoint expected
    return result[0] if result[0] < len(seq) else None

def run_simulation_binseg(N, q1, q2, num_iterations, max_tolerance):
    """
    Run multiple simulations of BinSeg changepoint detection and compute accuracy for each tolerance window.

    Parameters:
        N (int): Sequence length.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        num_iterations (int): Number of simulation runs.
        max_tolerance (int): Maximum tolerance window.

    Returns:
        list: Accuracy per tolerance window (list of floats).
    """
    buffer = smart_buffer(N)
    min_cp, max_cp = buffer, N - buffer
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    for tol in range(max_tolerance + 1):
        correct = 0
        for _ in range(num_iterations):
            cp = np.random.randint(min_cp, max_cp)
            seq = generate_sequence(N, cp, q1, q2)
            detected = detect_ruptures_binseg(seq)
            if detected is not None and abs(detected - cp) <= tol:
                correct += 1
        accuracy_per_tol.append(correct / num_iterations)
    return accuracy_per_tol

# =============================================================================
# Method 3: Ruptures PELT Algorithm with Custom Bernoulli Cost
# =============================================================================

def detect_ruptures_pelt(seq):
    """
    Detect changepoint using ruptures' PELT algorithm with a custom Bernoulli cost.

    Parameters:
        seq (np.ndarray): Binary sequence.

    Returns:
        int or None: Detected changepoint index, or None if not found.
    """
    model = CostBernoulli().fit(seq)
    algo = rpt.Pelt(custom_cost=model, min_size=1).fit(seq)
    penalty = 0.1 * np.log(len(seq))  # Penalty parameter for PELT
    result = algo.predict(pen=penalty)
    # Remove changepoints at the end (not valid)
    valid_cps = [cp for cp in result if cp < len(seq)]
    return valid_cps[0] if valid_cps else None

def run_simulation_pelt(N, q1, q2, num_iterations, max_tolerance):
    """
    Run multiple simulations of PELT changepoint detection and compute accuracy for each tolerance window.

    Parameters:
        N (int): Sequence length.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        num_iterations (int): Number of simulation runs.
        max_tolerance (int): Maximum tolerance window.

    Returns:
        list: Accuracy per tolerance window (list of floats).
    """
    buffer = smart_buffer(N)
    min_cp, max_cp = buffer, N - buffer
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    for tol in range(max_tolerance + 1):
        correct = 0
        for _ in range(num_iterations):
            cp = np.random.randint(min_cp, max_cp)
            seq = generate_sequence(N, cp, q1, q2)
            detected = detect_ruptures_pelt(seq)
            if detected is not None and abs(detected - cp) <= tol:
                correct += 1
        accuracy_per_tol.append(correct / num_iterations)
    return accuracy_per_tol

# =============================================================================
# Method 4: CUSUM (Cumulative Sum) Algorithm for Changepoint Detection
# =============================================================================

def detect_changepoint_cusum(seq, q1, q2):
    """
    Detect changepoint using the CUSUM algorithm for binary sequences.
    The CUSUM statistic is based on the log-likelihood ratio between two Bernoulli distributions.

    Parameters:
        seq (np.ndarray): Binary sequence.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.

    Returns:
        int: Detected changepoint index (location of maximum CUSUM statistic).
    """
    # Compute log-likelihood ratio for each observation
    llr_seq = np.array([
        np.log(q2 / q1) * x + np.log((1 - q2) / (1 - q1)) * (1 - x)
        for x in seq
    ])
    S = np.cumsum(llr_seq)  # Cumulative sum of LLRs
    G = S - np.minimum.accumulate(S)  # CUSUM statistic
    return np.argmax(G)  # Index of maximum CUSUM statistic

def run_simulation_cusum(N, q1, q2, num_iterations, max_tolerance):
    """
    Run multiple simulations of CUSUM changepoint detection and compute accuracy for each tolerance window.

    Parameters:
        N (int): Sequence length.
        q1 (float): Bernoulli parameter before changepoint.
        q2 (float): Bernoulli parameter after changepoint.
        num_iterations (int): Number of simulation runs.
        max_tolerance (int): Maximum tolerance window.

    Returns:
        list: Accuracy per tolerance window (list of floats).
    """
    buffer = smart_buffer(N)
    min_cp, max_cp = buffer, N - buffer
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    for tol in range(max_tolerance + 1):
        correct = 0
        for _ in range(num_iterations):
            cp = np.random.randint(min_cp, max_cp)
            seq = generate_sequence(N, cp, q1, q2)
            detected = detect_changepoint_cusum(seq, q1, q2)
            if abs(detected - cp) <= tol:
                correct += 1
        accuracy_per_tol.append(correct / num_iterations)
    return accuracy_per_tol

# =============================================================================
# Comparison Plot Type-I: Accuracy vs Sequence Length (N) for Fixed Tolerance
# =============================================================================

def compare_methods_accuracy_vs_N(methods, seq_lengths, w_h, epsilon, delta, num_iterations):
    """
    Compare the accuracy of different changepoint detection methods across sequence lengths,
    for a fixed tolerance window (delta).

    Parameters:
        methods (list): List of method names: ["mle", "binseg", "pelt", "cusum"]
        seq_lengths (list): List of sequence lengths to test.
        w_h (int): Hamming weight of vector h (used for computing q1).
        epsilon (float): BSC crossover probability.
        delta (int): Tolerance window for success.
        num_iterations (int): Number of simulation runs per sequence length.
    """
    # Compute Bernoulli parameters for before and after changepoint
    q1 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)
    q2 = 0.5

    # Map method names to simulation functions and display labels
    method_map = {
        "mle": (run_simulation_mle, "MLE"),
        "binseg": (run_simulation_binseg, "Binary Segmentation"),
        "pelt": (run_simulation_pelt, "PELT"),
        "cusum": (run_simulation_cusum, "CUSUM")
    }

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

    # Store results for terminal printing
    all_results = {}

    for color, method in zip(colors, methods):
        if method not in method_map:
            print(f"Unknown method '{method}' — skipping.")
            continue
        run_func, label = method_map[method]
        accuracies = []
        for N in seq_lengths:
            acc = run_func(N, q1, q2, num_iterations, max_tolerance=delta)
            accuracies.append(acc[delta])
        # Plot accuracy for this method
        plt.plot(seq_lengths, accuracies, marker='s', label=label, color=color)
        # Annotate each point with its value
        for x, y in zip(seq_lengths, accuracies):
            plt.text(x, y + 0.025, f"{y:.3f}", ha='center', va='bottom', fontsize=9, color=color)
        # Store for terminal output
        all_results[label] = accuracies

    plt.title(f"Accuracy vs Sequence Length (δ={delta}, ε={epsilon}, w={w_h})")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel(f"Accuracy at ±{delta}")
    plt.ylim(0, 1.05)
    plt.xticks(seq_lengths)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print accuracy values for each method and sequence length
    print("\nAccuracy values at tolerance ±{}:".format(delta))
    header = "Method".ljust(25) + "".join([f"N={N}".rjust(10) for N in seq_lengths])
    print(header)
    print("-" * len(header))
    for label, accuracies in all_results.items():
        row = label.ljust(25) + "".join([f"{acc:.3f}".rjust(10) for acc in accuracies])
        print(row)

# =============================================================================
# Comparison Plot Type-II: Accuracy vs BSC Parameter Epsilon for Fixed N and w
# =============================================================================

def compare_methods_accuracy_vs_epsilon(methods, epsilon_values, w_h, N, delta, num_iterations):
    """
    Compare the accuracy of different changepoint detection methods across BSC epsilon values,
    for a fixed sequence length (N) and Hamming weight (w_h).

    Parameters:
        methods (list): List of method names: ["mle", "binseg", "pelt", "cusum"]
        epsilon_values (list): List of BSC crossover probabilities (epsilon).
        w_h (int): Hamming weight of vector h (used to compute q1).
        N (int): Sequence length.
        delta (int): Tolerance window for success.
        num_iterations (int): Number of simulation runs per epsilon value.
    """
    q2 = 0.5
    method_map = {
        "mle": (run_simulation_mle, "MLE"),
        "binseg": (run_simulation_binseg, "Binary Segmentation"),
        "pelt": (run_simulation_pelt, "PELT"),
        "cusum": (run_simulation_cusum, "CUSUM")
    }

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(methods)))
    all_results = {}

    for color, method in zip(colors, methods):
        if method not in method_map:
            print(f"Unknown method '{method}' — skipping.")
            continue
        run_func, label = method_map[method]
        accuracies = []
        for epsilon in epsilon_values:
            q1 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)
            acc = run_func(N, q1, q2, num_iterations, max_tolerance=delta)
            accuracies.append(acc[delta])
        plt.plot(epsilon_values, accuracies, marker='s', label=label, color=color)
        for x, y in zip(epsilon_values, accuracies):
            plt.text(x, y + 0.025, f"{y:.3f}", ha='center', va='bottom', fontsize=9, color=color)
        all_results[label] = accuracies

    plt.title(f"Accuracy vs BSC Parameter ε (δ={delta}, N={N}, w={w_h})")
    plt.xlabel("BSC Parameter ε")
    plt.ylabel(f"Accuracy at ±{delta}")
    plt.ylim(0, 1.05)
    plt.xticks(epsilon_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print accuracy values for each method and epsilon value
    print(f"\nAccuracy values at tolerance ±{delta} for N={N}, w={w_h}:")
    header = "Method".ljust(25) + "".join([f"ε={e:.2f}".rjust(10) for e in epsilon_values])
    print(header)
    print("-" * len(header))
    for label, accuracies in all_results.items():
        row = label.ljust(25) + "".join([f"{acc:.3f}".rjust(10) for acc in accuracies])
        print(row)

# =============================================================================
# Plotting Utility: Run and Plot Results for a Single Method
# =============================================================================

def unique_method_plot(method="mle"):
    """
    Run the full simulation and plotting pipeline for the specified changepoint detection method.
    This function runs the simulation for a range of sequence lengths and plots accuracy vs tolerance.

    Parameters:
        method (str): Which method to use. One of "mle", "binseg", "pelt", "cusum".

    Raises:
        ValueError: If an unsupported method is specified.
    """
    # --- CONFIGURATION ---
    epsilon = 0.20        # BSC parameter (controls difference between q1 and q2)
    w_h = 4               # Hamming weight of vector h (affects q1)
    seq_lengths = [10, 15, 20, 50, 100, 200]  # Sequence lengths to test
    num_iterations = 1000  # Number of simulation runs per tolerance
    max_tolerance = 10     # Maximum tolerance window for accuracy

    # Compute Bernoulli parameters for before and after changepoint
    q1 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)
    q2 = 0.5  # After changepoint, always 0.5

    # Map method names to simulation functions and plot titles
    method_map = {
        "mle":   (run_simulation_mle,   "MLE Accuracy — Bernoulli Sequence"),
        "binseg":(run_simulation_binseg,"Ruptures (BinSeg) Accuracy — Bernoulli Sequence"),
        "pelt":  (run_simulation_pelt,  "Ruptures (PELT) Accuracy — Bernoulli Sequence"),
        "cusum": (run_simulation_cusum, "CUSUM Accuracy — Bernoulli Sequence"),
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from {list(method_map.keys())}.")

    run_func, title = method_map[method]
    # Run simulation for each sequence length and collect accuracy results
    all_accuracies = [
        run_func(N, q1, q2, num_iterations, max_tolerance)
        for N in seq_lengths
    ]
    # Plot and print results
    plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2, title)

# =============================================================================
# Example Usage of Unique Method Function 
# =============================================================================

# To run a specific method, uncomment the corresponding line below:
# unique_method_plot(method="mle")      # for MLE
# unique_method_plot(method="binseg")   # for Binary Segmentation
# unique_method_plot(method="pelt")     # for PELT
# unique_method_plot(method="cusum")    # for CUSUM

# =============================================================================
# Example Usage of Comparison Type-I: Accuracy vs Sequence Length (N) 
# =============================================================================
# compare_methods_accuracy_vs_N(
#     methods=["mle", "binseg", "pelt", "cusum"],
#     seq_lengths=[10, 15, 20, 50, 100, 200],
#     w_h=4,
#     epsilon=0.01,
#     delta=5, # tolerance window
#     num_iterations=1000
# )

# =============================================================================
# Example Usage of Comparison Type-II: Accuracy vs Epsilon (BSC Parameter) 
# =============================================================================
# compare_methods_accuracy_vs_epsilon(
#     methods=["mle", "binseg", "pelt", "cusum"],
#     epsilon_values=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
#     w_h=4,
#     N=100,
#     delta=5,
#     num_iterations=1000
# )
