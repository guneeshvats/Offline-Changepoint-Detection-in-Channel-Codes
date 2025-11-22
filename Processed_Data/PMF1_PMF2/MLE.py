import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# =============================================================================
# Common Utilities
# =============================================================================

def generate_categorical_sequence(N, changepoint, pmf1, pmf2):
    """
    Generate a categorical sequence of length N with a single changepoint.
    The first 'changepoint' samples are drawn from pmf1,
    and the remaining samples are drawn from pmf2.

    Parameters:
        N (int): Total length of the sequence.
        changepoint (int): Index at which the changepoint occurs (exclusive).
        pmf1 (list or np.ndarray): Probability mass function before changepoint.
        pmf2 (list or np.ndarray): Probability mass function after changepoint.

    Returns:
        np.ndarray: Categorical sequence of length N, with values in {0, ..., K-1}.
    """
    # Number of categories is inferred from the length of pmf1
    num_categories = len(pmf1)
    values = np.arange(num_categories)
    # Generate samples before the changepoint using pmf1
    before_cp = np.random.choice(values, size=changepoint, p=pmf1)
    # Generate samples after the changepoint using pmf2
    after_cp = np.random.choice(values, size=N - changepoint, p=pmf2)
    # Concatenate both segments to form the full sequence
    return np.concatenate([before_cp, after_cp])

def smart_buffer(N):
    """
    Compute a buffer size to avoid changepoints too close to the sequence edges.

    For short sequences (N <= 30), buffer is N//5 (at least 1).
    For longer sequences, buffer is fixed at 10.

    Parameters:
        N (int): Sequence length.

    Returns:
        int: Buffer size.
    """
    return max(1, N // 5) if N <= 30 else 10

def plot_accuracy(all_accuracies, max_tolerance, seq_lengths, title):
    """
    Plot detection accuracy vs. tolerance window for multiple sequence lengths.
    Also prints a table of accuracy values for each tolerance and sequence size.

    Parameters:
        all_accuracies (list of list of float): Each sublist contains accuracy values for a sequence length.
        max_tolerance (int): Maximum tolerance window to plot.
        seq_lengths (list of int): List of sequence lengths corresponding to all_accuracies.
        title (str): Title for the plot.
    """
    # Prepare table data for display
    table_data = []
    for acc_list, N in zip(all_accuracies, seq_lengths):
        for tol, acc in enumerate(acc_list):
            table_data.append({"Seq Length": N, "Tolerance": tol, "Accuracy": acc})
    df = pd.DataFrame(table_data)
    # Pivot the table for better readability
    pivot = df.pivot(index="Tolerance", columns="Seq Length", values="Accuracy")
    print("\nAccuracy Table (rows: Tolerance, columns: Sequence Length):")
    print(pivot.to_string(float_format="{:.4f}".format))
    print("-" * 60)

    # Plotting the accuracy curves for each sequence length
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
    plt.title(title)
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
# Method: MLE for Categorical Sequence
# =============================================================================

def compute_empirical_log_likelihood(segment):
    """
    Compute the log-likelihood of a categorical segment using empirical probabilities.

    The log-likelihood is computed as:
        sum_{i} n_i * log(n_i / n)
    where n_i is the count of category i, and n is the total number of samples.

    Parameters:
        segment (np.ndarray): 1D array of categorical values.

    Returns:
        float: Log-likelihood of the segment. Returns 0 if segment is empty.
    """
    n = len(segment)
    if n == 0:
        # Empty segment: log-likelihood is defined as 0
        return 0
    counts = Counter(segment)
    # Compute sum of n_i * log(n_i / n) for all categories present in the segment
    ll = sum(count * np.log(count / n) for count in counts.values())
    return ll

def detect_mle_categorical(seq, min_cp, max_cp):
    """
    Detect the changepoint in a categorical sequence using Maximum Likelihood Estimation (MLE).

    The changepoint is estimated by maximizing the sum of empirical log-likelihoods
    of the two segments (before and after the candidate changepoint).

    Parameters:
        seq (np.ndarray): 1D array of categorical values.
        min_cp (int): Minimum allowed changepoint index (inclusive).
        max_cp (int): Maximum allowed changepoint index (inclusive).

    Returns:
        int: Estimated changepoint index (tau) that maximizes the likelihood.
    """
    best_tau = None
    max_ll = -np.inf
    # Evaluate all possible changepoints in the allowed range
    for tau in range(min_cp, max_cp + 1):
        # Compute log-likelihood for the segment before the changepoint
        ll_before = compute_empirical_log_likelihood(seq[:tau])
        # Compute log-likelihood for the segment after the changepoint
        ll_after = compute_empirical_log_likelihood(seq[tau:])
        # Total log-likelihood for this changepoint
        total_ll = ll_before + ll_after
        # Update the best changepoint if this is the highest likelihood so far
        if total_ll > max_ll:
            max_ll = total_ll
            best_tau = tau
    return best_tau

def run_simulation_mle_categorical(N, pmf1, pmf2, num_iterations, max_tolerance):
    """
    Run multiple simulations of categorical MLE changepoint detection and compute accuracy.

    For each simulation:
        - Randomly select a changepoint within the allowed buffer.
        - Generate a categorical sequence with the changepoint.
        - Detect the changepoint using MLE.
        - For each tolerance window, record if detection is within tolerance.

    Parameters:
        N (int): Sequence length.
        pmf1 (list or np.ndarray): PMF before changepoint.
        pmf2 (list or np.ndarray): PMF after changepoint.
        num_iterations (int): Number of simulation runs.
        max_tolerance (int): Maximum tolerance window for accuracy calculation.

    Returns:
        list of float: Accuracy for each tolerance window (from 0 to max_tolerance).
    """
    buffer = smart_buffer(N)
    min_cp, max_cp = buffer, N - buffer
    # If the buffer is too large for the sequence, return zero accuracy
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    # For each tolerance window, compute the fraction of correct detections
    for tol in range(max_tolerance + 1):
        correct = 0
        for _ in range(num_iterations):
            # Randomly select a changepoint within the allowed range
            cp = np.random.randint(min_cp, max_cp)
            # Generate a sequence with the changepoint at cp
            seq = generate_categorical_sequence(N, cp, pmf1, pmf2)
            # Detect the changepoint using MLE
            detected = detect_mle_categorical(seq, min_cp, max_cp)
            # Check if the detected changepoint is within the tolerance window
            if abs(detected - cp) <= tol:
                correct += 1
        # Compute accuracy for this tolerance
        accuracy_per_tol.append(correct / num_iterations)
    return accuracy_per_tol

# =============================================================================
# Main Run Function
# =============================================================================

def run_all_mle_categorical():
    """
    Run and visualize MLE for categorical sequences across multiple sequence lengths.

    This function:
        - Defines a set of sequence lengths and PMFs for before/after changepoint.
        - Runs multiple simulations for each sequence length.
        - Plots and prints the detection accuracy as a function of tolerance window.
    """
    # Sequence lengths to evaluate
    seq_lengths = [10, 20, 30, 50, 100, 200]
    # Number of Monte Carlo simulations per sequence length
    num_iterations = 1000
    # Maximum tolerance window for accuracy calculation
    max_tolerance = 10
    # Probability mass functions before and after the changepoint
    pmf1 = [0.7, 0.2, 0.1]
    pmf2 = [0.2, 0.5, 0.3]

    # Run simulations for each sequence length and collect accuracy results
    all_accuracies = [
        run_simulation_mle_categorical(N, pmf1, pmf2, num_iterations, max_tolerance)
        for N in seq_lengths
    ]
    # Plot and print the results
    plot_accuracy(all_accuracies, max_tolerance, seq_lengths, "MLE — Categorical PMF Change Detection")

# Entry point for script execution
if __name__ == "__main__":
    run_all_mle_categorical()
