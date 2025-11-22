import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bernoulli

def generate_sequence(N, changepoint, q1, q2):
    """Generate a Bernoulli sequence with known changepoint and parameters."""
    before_cp = bernoulli.rvs(q1, size=changepoint)
    after_cp = bernoulli.rvs(q2, size=N - changepoint)
    return np.concatenate([before_cp, after_cp])

def log_likelihood(seq, tau, q1, q2):
    """Compute the total log-likelihood of the sequence with a changepoint at tau."""
    ll_before = np.sum(seq[:tau] * np.log(q1) + (1 - seq[:tau]) * np.log(1 - q1))
    ll_after = np.sum(seq[tau:] * np.log(q2) + (1 - seq[tau:]) * np.log(1 - q2))
    return ll_before + ll_after

def detect_changepoint(seq, q1, q2, min_cp, max_cp):
    """Detect changepoint by maximizing log-likelihood over all possible positions."""
    max_ll = -np.inf
    best_tau = None
    for tau in range(min_cp, max_cp + 1):
        ll = log_likelihood(seq, tau, q1, q2)
        if ll > max_ll:
            max_ll = ll
            best_tau = tau
    return best_tau

def smart_buffer(N):
    """Choose a smart buffer based on sequence length."""
    # if N <= 30:
    #     return max(1, N // 5)
    # else:
    #     return 10
    return 0

def run_simulation_mle(N, q1, q2, num_iterations, max_tolerance):
    """Run simulations using MLE and compute detection accuracy."""
    buffer = smart_buffer(N)
    min_cp = buffer
    max_cp = N - buffer
    if max_cp <= min_cp:
        return [0.0] * (max_tolerance + 1)

    accuracy_per_tol = []

    for tol in range(max_tolerance + 1):
        correct_detections = 0
        for _ in range(num_iterations):
            changepoint = np.random.randint(min_cp, max_cp)
            seq = generate_sequence(N, changepoint, q1, q2)
            detected = detect_changepoint(seq, q1, q2, min_cp=min_cp, max_cp=max_cp)
            if detected is not None and abs(detected - changepoint) <= tol:
                correct_detections += 1
        accuracy = correct_detections / num_iterations
        accuracy_per_tol.append(accuracy)

    return accuracy_per_tol

def plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2):
    """Plot accuracy vs tolerance window for multiple sequence lengths.
    Also print the accuracy values for each tolerance and sequence size in table form.
    """
    # Prepare table data
    table_data = []
    for idx, (accuracy_per_tol, N) in enumerate(zip(all_accuracies, seq_lengths)):
        for tol, acc in enumerate(accuracy_per_tol):
            table_data.append({"Seq Length": N, "Tolerance": tol, "Accuracy": acc})

    df = pd.DataFrame(table_data)
    table_pivot = df.pivot(index="Tolerance", columns="Seq Length", values="Accuracy")
    print("\nAccuracy Table (rows: Tolerance, columns: Sequence Length):")
    print(table_pivot.to_string(float_format="{:.4f}".format))
    print("-" * 60)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(seq_lengths)))
    for idx, (accuracy_per_tol, N) in enumerate(zip(all_accuracies, seq_lengths)):
        x_vals = list(range(max_tolerance + 1))
        plt.plot(
            x_vals,
            accuracy_per_tol,
            marker='o',
            color=colors[idx],
            label=f"Length={N}"
        )
        for x, y in zip(x_vals, accuracy_per_tol):
            plt.text(
                x, y + 0.02, f"{y:.2f}",
                ha='center', va='bottom', fontsize=8, color=colors[idx]
            )
    plt.title(f"MLE Accuracy — Bernoulli Sequence\nq1={q1:.3f}, q2={q2:.3f}")
    plt.xlabel("Tolerance Window")
    plt.ylabel("Detection Accuracy")
    plt.xticks(np.arange(0, max_tolerance + 1, 0.5))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    epsilon = 0.025      # BSC parameter [.001, .005, .01, .05, .08, .1, .15, .20]
    w_h = 4             # Weight of vector h
    seq_lengths = [5]# 7, 10, 15, 20, 50]  # List of sequence lengths to test
    num_iterations = 10000  # Iterations per tolerance level
    max_tolerance = 5     # Max tolerance window

    # --- COMPUTE q1 AND q2 ---
    q2 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)
    q1 = 0.5  # Always 0.5 after changepoint

    # --- RUN SIMULATION FOR EACH SEQUENCE LENGTH ---
    all_accuracies = []
    for N in seq_lengths:
        accuracy_per_tol = run_simulation_mle(N, q1, q2, num_iterations, max_tolerance)
        all_accuracies.append(accuracy_per_tol)

    # --- PLOT ---
    plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2)
