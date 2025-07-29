import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bernoulli

# --- Sequence Generation ---
def generate_sequence(N, changepoint, q1, q2):
    """
    Generate a binary sequence of length N with a single changepoint.
    The first 'changepoint' samples are drawn from Bernoulli(q1),
    and the remaining samples are drawn from Bernoulli(q2).
    """
    before_cp = bernoulli.rvs(q1, size=changepoint)
    after_cp = bernoulli.rvs(q2, size=N - changepoint)
    return np.concatenate([before_cp, after_cp])

# --- Smart Buffer ---
def smart_buffer(N):
    """
    Choose a buffer size to avoid changepoints too close to the sequence edges.
    For short sequences (N <= 30), use N//5 (at least 1).
    For longer sequences, use a fixed buffer of 10.
    """
    return max(1, N // 5) if N <= 30 else 10

# --- CUSUM Detection ---
def detect_changepoint_cusum(seq, q1, q2):
    """
    Detect the changepoint in a binary sequence using the CUSUM algorithm
    based on log-likelihood ratios for Bernoulli distributions.

    Args:
        seq: Binary sequence (numpy array or list).
        q1: Bernoulli parameter before changepoint.
        q2: Bernoulli parameter after changepoint.

    Returns:
        tau_hat: Estimated changepoint index (int).
    """
    # Compute the log-likelihood ratio for each sample
    llr_seq = np.array([
        np.log(q2 / q1) * x + np.log((1 - q2) / (1 - q1)) * (1 - x)
        for x in seq
    ])
    # Cumulative sum of LLRs
    S = np.cumsum(llr_seq)
    # Running minimum of the cumulative sum
    min_S = np.minimum.accumulate(S)
    # CUSUM statistic: difference between current S and its running minimum
    G = S - min_S
    # The changepoint is estimated as the index where G is maximized
    tau_hat = np.argmax(G)
    return tau_hat

# --- Simulation Framework ---
def run_simulation_cusum(N, q1, q2, num_iterations, max_tolerance):
    """
    Run multiple simulations to evaluate CUSUM changepoint detection accuracy.

    Args:
        N: Sequence length.
        q1: Bernoulli parameter before changepoint.
        q2: Bernoulli parameter after changepoint.
        num_iterations: Number of random trials per tolerance.
        max_tolerance: Maximum tolerance window for correct detection.

    Returns:
        accuracy_per_tol: List of detection accuracies for each tolerance window.
    """
    buffer = smart_buffer(N)
    min_cp = buffer
    max_cp = N - buffer
    # If the buffer is too large for the sequence, return zeros
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []
    # For each tolerance window, compute detection accuracy
    for tol in range(max_tolerance + 1):
        correct_detections = 0
        for _ in range(num_iterations):
            # Randomly select a changepoint within the valid range
            changepoint = np.random.randint(min_cp, max_cp)
            # Generate a sequence with the changepoint
            seq = generate_sequence(N, changepoint, q1, q2)
            # Detect the changepoint using CUSUM
            detected = detect_changepoint_cusum(seq, q1, q2)
            # Count as correct if detected changepoint is within tolerance
            if detected is not None and abs(detected - changepoint) <= tol:
                correct_detections += 1
        # Compute accuracy for this tolerance
        accuracy = correct_detections / num_iterations
        accuracy_per_tol.append(accuracy)
    return accuracy_per_tol

# --- Plotting ---
def plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2):
    """
    Plot detection accuracy vs. tolerance window for multiple sequence lengths.
    Also print a table of accuracy values for each tolerance and sequence length.

    Args:
        all_accuracies: List of accuracy lists (one per sequence length).
        max_tolerance: Maximum tolerance window.
        seq_lengths: List of sequence lengths.
        q1, q2: Bernoulli parameters.
    """
    # Prepare table data for display
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
    plt.title(f"CUSUM Accuracy — Bernoulli Sequence\nq1={q1:.3f}, q2={q2:.3f}")
    plt.xlabel("Tolerance Window")
    plt.ylabel("Detection Accuracy")
    plt.xticks(np.arange(0, max_tolerance + 1, 0.5))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main Block ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    epsilon = 0.20            # BSC parameter (crossover probability)
    w_h = 4                   # Hamming weight of vector h
    seq_lengths = [10, 15, 20, 50, 100, 200]  # Sequence lengths to test
    num_iterations = 1000     # Number of simulations per tolerance
    max_tolerance = 10        # Maximum tolerance window for accuracy

    # --- Compute Bernoulli parameters ---
    q1 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)  # Probability before changepoint
    q2 = 0.5                                     # Probability after changepoint

    # --- Run simulation for each sequence length ---
    all_accuracies = []
    for N in seq_lengths:
        accuracy = run_simulation_cusum(N, q1, q2, num_iterations, max_tolerance)
        all_accuracies.append(accuracy)

    # --- Plot the results and print the table ---
    plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2)
