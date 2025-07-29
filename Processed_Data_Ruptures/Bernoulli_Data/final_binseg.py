import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import ruptures as rpt
import pandas as pd

class CostBernoulli:
    """Custom cost function for Bernoulli-distributed binary sequences."""
    def fit(self, signal):
        self.signal = np.array(signal).astype(int)
        self.n = len(self.signal)
        return self

    def error(self, start, end):
        segment = self.signal[start:end]
        n = end - start
        if n == 0:
            return 0
        p_hat = np.mean(segment)
        if p_hat == 0 or p_hat == 1:
            return 0  # No variability
        n1 = np.sum(segment)
        n0 = n - n1
        return - (n1 * np.log(p_hat) + n0 * np.log(1 - p_hat))

    def cost(self, start, end):
        return self.error(start, end)

def generate_sequence(N, changepoint, q1, q2):
    """Generate a Bernoulli sequence with known changepoint and parameters."""
    before_cp = bernoulli.rvs(q1, size=changepoint)
    after_cp = bernoulli.rvs(q2, size=N - changepoint)
    return np.concatenate([before_cp, after_cp])

def smart_buffer(N):
    """Choose a smart buffer based on sequence length."""
    if N <= 30:
        return max(1, N // 5)
    else:
        return 10

def detect_with_ruptures(seq):
    """Use ruptures Binary Segmentation with custom Bernoulli cost to detect changepoint."""
    model = CostBernoulli().fit(seq)
    algo = rpt.Binseg(custom_cost=model).fit(seq)
    result = algo.predict(n_bkps=1)  # since we know there's one changepoint
    true_cps = [cp for cp in result if cp < len(seq)]
    if true_cps:
        return true_cps[0]
    else:
        return None



def run_simulation_ruptures(N, q1, q2, num_iterations, max_tolerance):
    """Run simulations using ruptures and compute detection accuracy."""
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
            detected = detect_with_ruptures(seq)
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

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(seq_lengths)))
    for idx, (accuracy_per_tol, N) in enumerate(zip(all_accuracies, seq_lengths)):
        x_vals = list(range(max_tolerance + 1))
        plt.plot(
            x_vals,
            accuracy_per_tol,
            marker='s',  # Use squares instead of circles
            color=colors[idx],
            label=f"Length={N}"
        )
        for x, y in zip(x_vals, accuracy_per_tol):
            plt.text(
                x, y + 0.02, f"{y:.2f}",
                ha='center', va='bottom', fontsize=8, color=colors[idx]
            )
    plt.title(f"Ruptures (BinSeg) Accuracy — Bernoulli Cost\nq1={q1:.3f}, q2={q2:.3f}")
    plt.xlabel("Tolerance Window")
    plt.ylabel("Detection Accuracy")
    plt.xticks(
        np.arange(0, max_tolerance + 1, 0.5)
    )  # More dense x-ticks, including in-between tolerance values
    plt.yticks(
        np.arange(0, 1.05, 0.05)
    )  # More dense y-ticks for accuracy
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    epsilon = 0.20        # BSC parameter
    w_h = 4              # Hamming weight of vector h
    seq_lengths = [10, 15, 20, 50, 100, 200]  # List of sequence lengths to test
    num_iterations = 1000  # Iterations per tolerance level
    max_tolerance = 10    # Maximum tolerance window

    # --- Compute Bernoulli parameters ---
    q1 = 0.5 - 0.5 * ((1 - 2 * epsilon) ** w_h)
    q2 = 0.5  # Always 0.5 after changepoint

    # --- Run simulation for each sequence length ---
    all_accuracies = []
    for N in seq_lengths:
        accuracy_per_tol = run_simulation_ruptures(N, q1, q2, num_iterations, max_tolerance)
        all_accuracies.append(accuracy_per_tol)

    # --- Plot the results ---
    plot_accuracy(all_accuracies, max_tolerance, seq_lengths, q1, q2)
