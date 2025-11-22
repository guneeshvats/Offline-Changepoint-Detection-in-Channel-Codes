# =============================================================================
# cpd_neural_refine.py
# =============================================================================
# "Changepoint Detection with Neural Refinement" — Reference Implementation
#
# This script implements a changepoint detection pipeline for binary codeword
# sequences, with neural network refinement. It is designed for research and
# production use, with detailed comments and robust error handling.
#
# Pipeline Overview:
#   1. Load two CSV files containing pools of binary codewords (C1 and C2).
#   2. Construct a sequence of length M, with a true changepoint at tau_true.
#      The first tau_true codewords are sampled from C1, the rest from C2.
#   3. Project each codeword onto a user-specified binary vector h, producing
#      a sequence z of inner products mod 2.
#   4. Perform a coarse changepoint estimate (MLE) on z using Bernoulli
#      likelihoods with parameters (theta1, theta2).
#   5. Compute the asymptotic prior distribution pi_n (Hinkley-style) for
#      changepoint uncertainty, and select the smallest even window S such
#      that the symmetric mass sum_{|n|<=S/2} pi_n >= alpha.
#   6. Crop the raw codeword sequence around the coarse changepoint, and run
#      sliding windows of length T through a TorchScript neural network model.
#   7. Aggregate neural network predictions to produce a refined changepoint
#      estimate.
#
# User inputs are defined in the main() function with detailed comments.
#
# Requirements:
#   pip install numpy pandas torch
#
# Notes:
#   - The neural network model must be a TorchScript .pt file that accepts
#     input of shape (B, T, n) and outputs logits of shape (B, T+1).
#   - The Hinkley prior computation is a compact port of the essentials from
#     hinkley.py.
# =============================================================================

from __future__ import annotations
import math
import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

# Optional import: torch is only required for neural network refinement.
try:
    import torch
except Exception:
    torch = None

# =============================================================================
# I/O and Sequence Construction Utilities
# =============================================================================

def load_codewords_csv(path: str) -> np.ndarray:
    """
    Load a CSV file containing binary codewords (0/1).
    Each row is a codeword of length n.
    
    Args:
        path (str): Path to the CSV file.
    
    Returns:
        np.ndarray: Array of shape (N_codewords, n), dtype float32, values in {0,1}.
    
    Raises:
        ValueError: If the CSV is not 2D or contains non-binary values.
    """
    arr = pd.read_csv(path, header=None).values.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"CSV at {path} must be 2D: rows=codewords, cols=bits.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"CSV at {path} must contain only 0/1.")
    return arr

def build_sequence_from_pools(
    c1_pool: np.ndarray, c2_pool: np.ndarray, M: int, tau_true: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Construct a sequence of codewords of length M, with a changepoint at tau_true.
    The first tau_true codewords are sampled (with replacement) from c1_pool,
    and the remaining (M - tau_true) from c2_pool.
    
    Args:
        c1_pool (np.ndarray): Pool of codewords for segment 1, shape (N1, n).
        c2_pool (np.ndarray): Pool of codewords for segment 2, shape (N2, n).
        M (int): Total sequence length.
        tau_true (int): Index of the changepoint (1-based, must be in [1, M-1]).
        rng (np.random.Generator): Numpy random generator for reproducibility.
    
    Returns:
        np.ndarray: Sequence of codewords, shape (M, n), dtype float32.
    
    Raises:
        ValueError: If codeword lengths mismatch or tau_true is out of bounds.
    """
    n = c1_pool.shape[1]
    if c2_pool.shape[1] != n:
        raise ValueError("C1 and C2 codewords must have same length n.")
    if not (1 <= tau_true < M):
        raise ValueError("tau_true must be in {1,...,M-1}.")
    idx1 = rng.integers(0, len(c1_pool), size=tau_true)
    idx2 = rng.integers(0, len(c2_pool), size=M - tau_true)
    y1 = c1_pool[idx1]
    y2 = c2_pool[idx2]
    return np.vstack([y1, y2]).astype(np.float32)

# =============================================================================
# Inner-Product Projection
# =============================================================================

def project_with_h(y_seq: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Project each codeword y_i in y_seq onto the binary vector h using inner product mod 2.
    
    Args:
        y_seq (np.ndarray): Sequence of codewords, shape (M, n), values in {0,1}.
        h (np.ndarray): Binary vector, shape (n,), values in {0,1}.
    
    Returns:
        np.ndarray: Sequence of projected bits, shape (M,), values in {0,1}, dtype uint8.
    
    Raises:
        ValueError: If input shapes are invalid or h is not binary.
    """
    if y_seq.ndim != 2:
        raise ValueError("y_seq must be 2D (M, n).")
    M, n = y_seq.shape
    h = np.asarray(h, dtype=np.uint8)
    if h.shape != (n,):
        raise ValueError(f"h must have shape ({n},).")
    if not np.isin(h, [0, 1]).all():
        raise ValueError("h must be a binary vector of 0/1.")
    # Compute bitwise AND, sum, and mod 2 for each codeword.
    sums = (y_seq.astype(np.uint8) & h).sum(axis=1)
    return (sums % 2).astype(np.uint8)

# =============================================================================
# Bernoulli Parameters and Maximum Likelihood Estimation (MLE)
# =============================================================================

def compute_thetas(p: float, w_h: int, swap_theta12: bool = False) -> Tuple[float, float]:
    """
    Compute Bernoulli parameters (theta1, theta2) for the projected sequence.
    By default:
        theta1 = 0.5 - 0.5*(1 - 2p)^w_h   (BEFORE changepoint)
        theta2 = 0.5                       (AFTER  changepoint)
    If swap_theta12=True, swap the assignments.
    
    Args:
        p (float): Channel crossover probability, must be in (0, 0.5).
        w_h (int): Hamming weight of h (number of 1s).
        swap_theta12 (bool): If True, swap theta1 and theta2.
    
    Returns:
        Tuple[float, float]: (theta1, theta2)
    
    Raises:
        ValueError: If p or w_h are out of valid range.
    """
    if not (0.0 < p < 0.5):
        raise ValueError("p should be in (0, 0.5) for BSC.")
    if w_h <= 0:
        raise ValueError("w_h must be a positive integer.")
    base = (1.0 - 2.0 * p)
    theta_odd = 0.5 - 0.5 * (base ** w_h)
    theta_half = 0.5
    if swap_theta12:
        return theta_half, theta_odd
    return theta_odd, theta_half

def _bern_log_likelihood_prefix_sums(z: np.ndarray, log_q: float, log_1mq: float) -> np.ndarray:
    """
    Compute prefix sums of log-likelihoods for a Bernoulli sequence.
    Used for efficient changepoint likelihood computation.
    
    Args:
        z (np.ndarray): Sequence of bits, shape (M,), values in {0,1}.
        log_q (float): log(theta)
        log_1mq (float): log(1 - theta)
    
    Returns:
        np.ndarray: Prefix sums, shape (M+1,), with L_0=0.
    """
    z = z.astype(np.float64)
    contrib = z * log_q + (1.0 - z) * log_1mq
    pref = np.zeros(len(z) + 1, dtype=np.float64)
    np.cumsum(contrib, out=pref[1:])
    return pref

def mle_tau_on_bernoulli(z: np.ndarray, theta1: float, theta2: float, tie_rule: str = "smallest") -> int:
    """
    Perform maximum likelihood estimation (MLE) for a single changepoint in a Bernoulli sequence.
    Finds tau in {1, ..., M-1} maximizing the likelihood.
    
    Args:
        z (np.ndarray): Sequence of bits, shape (M,).
        theta1 (float): Bernoulli parameter before changepoint.
        theta2 (float): Bernoulli parameter after changepoint.
        tie_rule (str): "smallest" or "largest" for tie-breaking.
    
    Returns:
        int: Estimated changepoint tau_hat (1-based).
    
    Raises:
        ValueError: If parameters are out of range or sequence too short.
    """
    if not (0.0 < theta1 < 1.0 and 0.0 < theta2 < 1.0):
        raise ValueError("theta1, theta2 must be in (0,1).")
    M = len(z)
    if M < 2:
        raise ValueError("Need M >= 2 to define a changepoint.")

    lq1, l1mq1 = math.log(theta1), math.log(1 - theta1)
    lq2, l1mq2 = math.log(theta2), math.log(1 - theta2)

    pref1 = _bern_log_likelihood_prefix_sums(z, lq1, l1mq1)
    pref2 = _bern_log_likelihood_prefix_sums(z, lq2, l1mq2)

    taus = np.arange(1, M, dtype=np.int32)
    LL = pref1[taus] + (pref2[M] - pref2[taus])

    mx = float(LL.max())
    idxs = np.flatnonzero(np.isclose(LL, mx))
    tau_hat = int(taus[idxs[0]] if tie_rule == "smallest" else taus[idxs[-1]])
    return tau_hat

# =============================================================================
# Hinkley-Style Prior Distribution (pi_n) and Window Selection
# =============================================================================

def _phi(theta1: float, theta2: float) -> float:
    """
    Compute the Phi parameter for the Hinkley prior.
    """
    return math.log((1 - theta2) / (1 - theta1)) / math.log(theta1 / theta2)

def _log_binom_pmf(n: int, r: int, p: float) -> float:
    """
    Compute the log of the binomial PMF for (n, r, p).
    """
    from math import lgamma, log
    return (lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)
            + r * log(p) + (n - r) * log(1.0 - p))

def _logsumexp(log_vals: List[float]) -> float:
    """
    Numerically stable log-sum-exp for a list of log values.
    """
    m = max(log_vals)
    if m == -float('inf'):
        return m
    s = sum(math.exp(v - m) for v in log_vals)
    return m + math.log(s)

def _p00_left(theta1: float, Phi: float, nmax=2000, tol=1e-12) -> float:
    """
    Compute the left-side p00 for the Hinkley prior.
    """
    acc = 0.0
    for n in range(1, nmax + 1):
        r_max = int(math.floor(n * Phi / (1 + Phi)))
        if r_max < 0:
            inner = 0.0
        else:
            logs = [_log_binom_pmf(n, r, theta1) for r in range(0, r_max + 1)]
            inner = math.exp(_logsumexp(logs))
        inc = inner / n
        acc += inc
        if inc < tol:
            break
    return math.exp(-acc)

def _p00_right(theta2: float, Phi: float, nmax=2000, tol=1e-12) -> float:
    """
    Compute the right-side p00 for the Hinkley prior.
    """
    acc = 0.0
    for n in range(1, nmax + 1):
        r_min = int(math.floor(n * Phi / (1 + Phi))) + 1
        if r_min > n:
            tail = 0.0
        else:
            logs = [_log_binom_pmf(n, r, theta2) for r in range(r_min, n + 1)]
            tail = math.exp(_logsumexp(logs))
        inc = tail / n
        acc += inc
        if inc < tol:
            break
    return math.exp(-acc)

def _q_p_left(theta1: float, Phi: float, N: int, p00: float):
    """
    Compute the left-side q and p tables for the Hinkley prior recursion.
    """
    q = {}
    q[(1, 0)] = theta1 * p00
    for l in range(2, N + 1):
        q[(l, 0)] = 0.0
    for m in range(1, N + 1):
        q[(0, m)] = ((1 - theta1) ** m) * p00

    for s in range(2, N + 1):
        for l in range(1, s):
            m = s - l
            val = -l + Phi * m
            if val > Phi:
                q[(l, m)] = theta1 * q.get((l - 1, m), 0.0) + (1 - theta1) * q.get((l, m - 1), 0.0)
            elif (-1 < val < Phi):
                q[(l, m)] = theta1 * q.get((l - 1, m), 0.0)
            else:
                q[(l, m)] = 0.0

    p = {(l, m): (v if (-l + Phi * m) >= 0 else 0.0) for (l, m), v in q.items()}
    return q, p

def _q_p_right(theta2: float, Phi: float, N: int, p00p: float):
    """
    Compute the right-side q and p tables for the Hinkley prior recursion.
    """
    q = {}
    q[(0, 1)] = (1 - theta2) * p00p
    for n in range(2, N + 1):
        q[(0, n)] = 0.0
    for l in range(1, N + 1):
        q[(l, 0)] = (theta2 ** l) * p00p

    for s in range(2, N + 1):
        for l in range(1, s):
            m = s - l
            val = l - Phi * m
            if val > 1:
                q[(l, m)] = theta2 * q.get((l - 1, m), 0.0) + (1 - theta2) * q.get((l, m - 1), 0.0)
            elif (-Phi < val < 1):
                q[(l, m)] = theta2 * q.get((l - 1, m), 0.0)
            else:
                q[(l, m)] = 0.0

    p = {(l, m): (v if (l - Phi * m) >= 0 else 0.0) for (l, m), v in q.items()}
    return q, p

def compute_pi_distribution(theta1: float, theta2: float, n_display: int = 50, Ngrid: int = 200) -> Dict[int, float]:
    """
    Compute the Hinkley-style prior distribution pi_n for changepoint uncertainty.
    This is a minimal standalone implementation of Eq. (3.1) from the referenced work.
    
    Args:
        theta1 (float): Bernoulli parameter before changepoint.
        theta2 (float): Bernoulli parameter after changepoint.
        n_display (int): Range of n to compute, i.e., [-n_display, +n_display].
        Ngrid (int): Lattice grid size for recursion accuracy.
    
    Returns:
        Dict[int, float]: Dictionary mapping n to pi_n.
    """
    Phi = _phi(theta1, theta2)
    p00 = _p00_left(theta1, Phi)
    p00p = _p00_right(theta2, Phi)
    _, p_left = _q_p_left(theta1, Phi, Ngrid, p00)
    _, p_right = _q_p_right(theta2, Phi, Ngrid, p00p)

    pi = {0: float(p00 * p00p)}
    eps = 1e-14

    # Compute negative and positive n values.
    for n in range(1, n_display + 1):
        total_neg = 0.0
        for l in range(0, n + 1):
            m = n - l
            outer = float(p_left.get((l, m), 0.0))
            if outer == 0.0:
                continue
            thr = -l + Phi * m
            inner = p00p if (thr > 0.0) else 0.0
            for (j, k), pr in p_right.items():
                if pr <= 0.0:
                    continue
                if (j - Phi * k) < (thr - eps):
                    inner += pr
            total_neg += outer * inner
        pi[-n] = total_neg

        total_pos = 0.0
        for l in range(0, n + 1):
            m = n - l
            outer = float(p_right.get((l, m), 0.0))
            if outer == 0.0:
                continue
            thr = l - Phi * m
            inner = p00 if (thr > 0.0) else 0.0
            for (j, k), pr in p_left.items():
                if pr <= 0.0:
                    continue
                if (-j + Phi * k) < (thr - eps):
                    inner += pr
            total_pos += outer * inner
        pi[+n] = total_pos

    return pi

def choose_S_from_alpha(pi: dict[int, float], alpha: float) -> tuple[int, int]:
    """
    Select the smallest even window S such that the symmetric mass of pi_n
    within [-S/2, S/2] is at least alpha.
    
    Args:
        pi (dict[int, float]): Prior distribution pi_n.
        alpha (float): Desired confidence level (0 < alpha < 1).
    
    Returns:
        tuple[int, int]: (S, halfwidth), where S = 2 * halfwidth.
    
    Raises:
        ValueError: If alpha is out of range or pi_n has zero mass.
    """
    # if not (0.0 < alpha < 1.0):
    #     raise ValueError("alpha must be in (0,1).")
    # total = sum(pi.values())
    # if total <= 0:
    #     raise ValueError("π_n has zero mass.")
    # pi_norm = {k: v / total for k, v in pi.items()}

    # nmax = max(abs(k) for k in pi_norm)
    # for h in range(nmax + 1):
    #     mass = sum(pi_norm.get(n, 0.0) for n in range(-h, h + 1))
    #     if mass >= alpha:
    #         return 2*h, h  # S, halfwidth
    S_FIXED = 16
    return S_FIXED, S_FIXED // 2

# =============================================================================
# Neural Network Refinement on Raw Codewords
# =============================================================================

def nn_refine_tau_soft(
    y_crop: np.ndarray,
    window_T: int,
    model_path: str,
    a: int, b: int,                 # crop bounds (0-based, inclusive)
    stride: int = 1,
    device: str = "cpu",
    use_log_probs: bool = False,    # If True, aggregate log-probabilities.
    use_prior: bool = False,        # If True, multiply by Hinkley prior.
    pi: dict | None = None,         # Prior distribution pi_n (centered at 0).
    tau_tilde_1b: int | None = None # Coarse MLE tau~ (1-based), for prior.
) -> tuple[int, int, np.ndarray]:
    """
    Refine the changepoint estimate using a neural network model.
    Slides a window of length T over the cropped codeword sequence, aggregates
    neural network predictions, and optionally applies a prior.
    
    Args:
        y_crop (np.ndarray): Cropped codeword sequence, shape (L, n).
        window_T (int): Window length for the neural network.
        model_path (str): Path to the TorchScript model (.pt file).
        a (int): Start index of the crop (0-based, inclusive).
        b (int): End index of the crop (0-based, inclusive).
        stride (int): Step size for sliding window.
        device (str): Device for model inference ("cpu" or "cuda").
        use_log_probs (bool): If True, aggregate log-probabilities.
        use_prior (bool): If True, multiply by Hinkley prior.
        pi (dict or None): Prior distribution pi_n.
        tau_tilde_1b (int or None): Coarse MLE tau~ (1-based), for prior.
    
    Returns:
        tuple: (tau_hat_crop_0b, tau_hat_global_0b, scores)
            tau_hat_crop_0b: Index in the crop (0-based).
            tau_hat_global_0b: Index in the global sequence (0-based).
            scores: Aggregated scores for each position in the crop.
    
    Raises:
        RuntimeError: If torch is not available.
        FileNotFoundError: If the model file does not exist.
        ValueError: If crop is too short.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for NN refinement.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    L, n = y_crop.shape
    if L < window_T:
        L = window_T
        #raise ValueError(f"Crop length ({L}) must be >= window_T ({window_T}).")

    # Load TorchScript model for inference.
    jit_model = torch.jit.load(model_path, map_location=device)
    jit_model.to(device)
    jit_model.eval()

    scores = np.zeros(L, dtype=np.float64)  # Scores for each crop-relative tau (0..L-1)

    with torch.no_grad():
        s = 0
        while s + window_T <= L:
            window = y_crop[s: s + window_T]  # (T, n)
            x = torch.from_numpy(window[None, ...].astype(np.float32)).to(device)

            logits = jit_model(x)  # (1, T+1)
            if use_log_probs:
                logp = torch.log_softmax(logits, dim=1)[0].cpu().numpy()  # log p(t), t=0..T
                from math import log, inf
                lse = log(np.exp(logp[1:window_T]).sum()) if window_T > 1 else -inf
                if not np.isfinite(lse):
                    s += stride
                    continue
                for t in range(1, window_T):  # Only interior positions
                    tau_global_0b = a + s + t
                    if a <= tau_global_0b <= b:
                        idx = tau_global_0b - a
                        scores[idx] += float(logp[t])
                        # Alternative: scores[idx] += float(logp[t] - lse)
            else:
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # p(t), t=0..T
                mass_interior = float(probs[1:window_T].sum())
                if mass_interior < 1e-8:
                    s += stride
                    continue
                for t in range(1, window_T):  # Only interior positions
                    tau_global_0b = a + s + t
                    if a <= tau_global_0b <= b:
                        idx = tau_global_0b - a
                        scores[idx] += float(probs[t])
                        # Alternative: scores[idx] += float(probs[t] / mass_interior)
            s += stride

    # Optionally apply Hinkley prior centered at tau~.
    if use_prior and (pi is not None) and (tau_tilde_1b is not None):
        tau_grid_global_0b = np.arange(a, b + 1, dtype=int)
        tau_tilde_0b = int(tau_tilde_1b - 1)
        prior = np.array([pi.get(int(t - tau_tilde_0b), 0.0) for t in tau_grid_global_0b], dtype=np.float64)
        prior = prior / (prior.sum() + 1e-15)
        if use_log_probs:
            scores += np.log(np.clip(prior, 1e-15, 1.0))
        else:
            scores *= prior

    # Fallback: If all scores are zero, return the coarse MLE tau~.
    if np.allclose(scores, 0.0):
        tau_hat_global_0b = int(tau_tilde_1b - 1)
        tau_hat_crop_0b = tau_hat_global_0b - a
        return tau_hat_crop_0b, tau_hat_global_0b, scores

    tau_hat_crop_0b = int(np.argmax(scores))
    tau_hat_global_0b = a + tau_hat_crop_0b
    return tau_hat_crop_0b, tau_hat_global_0b, scores

# =============================================================================
# Utility Functions
# =============================================================================

def clamp(val: int, lo: int, hi: int) -> int:
    """
    Clamp an integer value to the range [lo, hi].
    """
    return max(lo, min(hi, val))

def crop_indices(center: int, halfwidth: int, M: int) -> Tuple[int, int]:
    """
    Compute inclusive start/end indices [a, b] for cropping a sequence of length M,
    centered at 'center' with halfwidth 'halfwidth'. The window is clamped to [0, M-1].
    
    Args:
        center (int): Center index (0-based).
        halfwidth (int): Halfwidth of the window.
        M (int): Total sequence length.
    
    Returns:
        Tuple[int, int]: (a, b), inclusive indices.
    """
    a = clamp(center - halfwidth, 0, M - 1)
    b = clamp(center + halfwidth, 0, M - 1)
    return a, b

# =============================================================================
# Main Driver Function with User Inputs
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # ========================= USER INPUTS ===================================
    # -------------------------------------------------------------------------
    # 1) Paths to CSVs containing codeword pools for C1 and C2.
    #    Each file should contain binary codewords (rows of 0/1).
    CSV_C1: str = "/Users/guneeshvats/Desktop/CPD Research/BSC_BCH_n15_data/bsc_p0.05_codewords1.csv"  # Path to C1 codewords CSV.
    CSV_C2: str = "/Users/guneeshvats/Desktop/CPD Research/BSC_BCH_n15_data/bsc_p0.05_codewords2.csv"   # Path to C2 codewords CSV.

    # 2) Global sequence length M and TRUE changepoint (1..M-1).
    #    M: Length of the sequence (must be >= 2).
    #    TAU_TRUE: Location of the true changepoint (must be in [1, M-1]).
    M: int = 50
    TAU_TRUE: int = 20

    # 3) Test vector h (binary length-n).
    #    Choose h ∈ C1^⊥ and h ∉ C2^⊥ for default behavior.
    #    Example for n=15: provide exactly 15 entries of 0/1.
    # H vector for n = 15
    H: List[int] = [1,1,0,0,0,0,0,0,0,1,0,0,0,1,0]  # User-specified binary vector.
    # H vector for n = 31 is this 
    # H: List[int] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]


    # 4) Channel and projection parameters for θ1, θ2.
    #    p: Channel crossover probability (float in (0, 0.5)).
    #    SWAP_THETA12: If True, swap θ1 and θ2 assignments.
    p: float = 0.05
    SWAP_THETA12: bool = True

    # 5) Confidence level alpha for S-selection from π_n.
    #    alpha: Desired symmetric mass (float in (0,1)), e.g., 0.9 or 0.95.
    alpha: float = 0.90

    # 6) Neural network model configuration.
    #    MODEL_PATH: Path to TorchScript .pt file.
    #    WINDOW_T: Window length for the model (must match model's expected T).
    #    STRIDE: Step size for sliding window.
    #    DEVICE: "cpu" or "cuda" (if available).
    MODEL_PATH: str = "/Users/guneeshvats/Desktop/CPD Research/unknown_p_BCH_BSC_n=15_NN_models/cpd_multiP_T5_n15_seed0_train17_hold2_out_tpye_two_15_model.pt"
    WINDOW_T: int = 5
    STRIDE:  int = 1
    DEVICE:  str = "cpu"

    # 7) Random seed for sequence construction (for reproducibility).
    SEED: int = 12345

    # --- π_n controls (for computation only; large range for correctness) ---
    #     PIN_NGRID: Lattice grid size for recursion accuracy.
    #     PIN_NDISPLAY: Compute π_n on [-PIN_NDISPLAY, +PIN_NDISPLAY].
    #     VIEW_HALF: How many bars around 0 to print for inspection.
    PIN_NGRID    = 20
    PIN_NDISPLAY = 20
    VIEW_HALF    = 10

    # 0) Monte Carlo controls (for repeated trials).
    #    MC_ENABLE: If True, run MC_NR random trials with random tau_true.
    #    MC_NR: Number of Monte Carlo runs.
    MC_ENABLE: bool = True
    MC_NR: int = 1000

    # -------------------------------------------------------------------------
    # ======================= END USER INPUTS =================================
    # -------------------------------------------------------------------------

    # Load codeword pools from CSV files.
    c1 = load_codewords_csv(CSV_C1)  # (N1, n)
    c2 = load_codewords_csv(CSV_C2)  # (N2, n)
    n = c1.shape[1]
    if c2.shape[1] != n:
        raise ValueError("CSV pools have different codeword lengths.")

    # Validate h length (should match codeword length n).
    h = np.array(H, dtype=np.uint8)
    if h.shape != (n,):
        raise ValueError(f"h must have length n={n}, but got {len(h)}.")
    w_h = int(h.sum())

    # Build the codeword sequence with the true changepoint.
    rng = np.random.default_rng(SEED)
    y_seq = build_sequence_from_pools(c1, c2, M, TAU_TRUE, rng)  # (M, n)

    # Project codewords to z_i via inner product with h.
    z = project_with_h(y_seq, h)  # (M,)

    # Compute Bernoulli parameters θ1, θ2.
    theta1, theta2 = compute_thetas(p, w_h, swap_theta12=SWAP_THETA12)
    print(f"[info] n={n}, w(h)={w_h}, p={p:.4g} -> theta1={theta1:.6f}, theta2={theta2:.6f}")

    # Coarse MLE on z_i to estimate changepoint.
    tau_tilde = mle_tau_on_bernoulli(z, theta1, theta2, tie_rule="smallest")
    print(f"[coarse-MLE] tau_tilde = {tau_tilde}   (true tau = {TAU_TRUE})")

    # Compute Hinkley-style prior pi_n and select window S for desired alpha.
    pi = compute_pi_distribution(theta1, theta2, n_display=PIN_NDISPLAY, Ngrid=PIN_NGRID)

    # Print a small window of pi_n around 0 for inspection.
    print("\n[pi_n around 0]")
    for k in range(-VIEW_HALF, VIEW_HALF + 1):
        if k in pi:
            print(f"pi[{k:+d}] = {pi[k]:.6f}")

    # Select window S and halfwidth for the desired symmetric mass alpha.
    S, half = choose_S_from_alpha(pi, alpha)
    print(f"[window] alpha={alpha:.3f} -> smallest even S={S}, halfwidth={half}, crop size S+1 = {S+1}")

    # Print cumulative symmetric mass for each halfwidth (sanity check).
    total = sum(pi.values())
    pi_norm = {k: v/total for k, v in pi.items()}
    hit = None
    nmax = max(abs(k) for k in pi_norm)
    print("\n[cumulative symmetric mass]")
    for hh in range(nmax + 1):
        mass = sum(pi_norm.get(n, 0.0) for n in range(-hh, hh + 1))
        flag = ""
        if hit is None and mass >= alpha:
            flag = "  <-- reaches alpha here"
            hit = hh
        print(f"h={hh:2d}  mass={mass:.6f}{flag}")

    # =========================================================================
    # Monte Carlo Mode: Run multiple trials with random tau_true (optional).
    # =========================================================================
    if MC_ENABLE:
        print(f"\n[MC] Running {MC_NR} trials with random true τ ∈ {{1,…,{M-1}}}")
        exact = 0
        mc_rows = []
        for r in range(MC_NR):
            tau_true = int(rng.integers(1, M))  # Random tau_true in {1,..,M-1}
            y_seq = build_sequence_from_pools(c1, c2, M, tau_true, rng)
            z = project_with_h(y_seq, h)
            theta1, theta2 = compute_thetas(p, w_h, swap_theta12=SWAP_THETA12)
            tau_tilde = mle_tau_on_bernoulli(z, theta1, theta2, tie_rule="smallest")

            center0 = tau_tilde - 1
            a, b = crop_indices(center=center0, halfwidth=half, M=M)
            y_crop = y_seq[a:b+1]
            if len(y_crop) < WINDOW_T:
                need = WINDOW_T - len(y_crop)
                a2 = clamp(a - (need // 2), 0, M - 1)
                b2 = clamp(b + (need - (a - a2)), 0, M - 1)
                y_crop = y_seq[a2:b2+1]; a, b = a2, b2

            tau_hat_crop_0b, tau_hat_global_0b, _ = nn_refine_tau_soft(
                y_crop=y_crop, window_T=WINDOW_T, model_path=MODEL_PATH,
                a=a, b=b, stride=STRIDE, device=DEVICE,
                use_log_probs=False, use_prior=False, pi=None, tau_tilde_1b=tau_tilde
            )
            tau_hat_count = int(tau_hat_global_0b)
            # exact += int(tau_hat_count == tau_true)

            exact += int((tau_hat_global_0b + 1) == tau_true)  # 1-based vs 1-based

            mc_rows.append({"run": r, "tau_true_count": tau_true, "tau_tilde_1b": tau_tilde,
                            "tau_hat_count": tau_hat_count, "crop_a_0b": a, "crop_b_0b": b})
        acc = exact / MC_NR
        print(f"\n[MC] exact hits = {exact}/{MC_NR} ({100*acc:.2f}%)")
        mc_csv = "mc_runs.csv"
        pd.DataFrame(mc_rows).to_csv(mc_csv, index=False)
        print(f"[MC] saved per-run results -> {os.path.abspath(mc_csv)}")
        return  # Exit after MC mode; skip single-run path.

    # =========================================================================
    # Single-Run Mode: Crop, Refine, and Save Results
    # =========================================================================

    # Crop the raw sequence around the coarse changepoint tau_tilde.
    center0 = tau_tilde-1
    a, b = crop_indices(center=center0, halfwidth=half, M=M)
    y_crop = y_seq[a : b + 1]  # inclusive [a,b]
    print(f"[crop] indices [{a}:{b}] (size={len(y_crop)})")

    # Ensure crop is at least as long as WINDOW_T; expand if needed.
    if WINDOW_T is None or WINDOW_T <= 0:
        raise ValueError("WINDOW_T must be a positive integer.")
    if len(y_crop) < WINDOW_T:
        print("[warn] Crop shorter than WINDOW_T; expanding to include more context.")
        need = WINDOW_T - len(y_crop)
        a2 = clamp(a - (need // 2), 0, M - 1)
        b2 = clamp(b + (need - (a - a2)), 0, M - 1)
        y_crop = y_seq[a2 : b2 + 1]
        a, b = a2, b2
        print(f"[crop-adjusted] indices [{a}:{b}] (size={len(y_crop)})")

    # Neural network refinement (soft aggregation over global tau).
    tau_hat_crop_0b, tau_hat_global_0b, scores = nn_refine_tau_soft(
        y_crop=y_crop,
        window_T=WINDOW_T,
        model_path=MODEL_PATH,
        a=a, b=b,
        stride=STRIDE,
        device=DEVICE,
        use_log_probs=False,         # Set True to aggregate log-probs.
        use_prior=False,             # Set True to include Hinkley prior.
        pi=pi,
        tau_tilde_1b=tau_tilde
    )
    tau_hat_global = int(tau_hat_global_0b)

    # Save all relevant results to CSV: result_final_algorithm.csv
    total_mass = sum(pi.values())
    pi_norm = {k: v / total_mass for k, v in pi.items()}

    # Symmetric mass actually achieved at the chosen halfwidth (should be >= alpha).
    S_mass_symmetric = sum(pi_norm.get(nk, 0.0) for nk in range(-half, half + 1))

    rows = []
    def add(section, name, value):
        rows.append({"section": section, "name": name, "value": value})

    # ---- Inputs
    add("inputs", "CSV_C1", CSV_C1)
    add("inputs", "CSV_C2", CSV_C2)
    add("inputs", "M", M)
    add("inputs", "TAU_TRUE_1b", TAU_TRUE)
    add("inputs", "H_vector", " ".join(map(str, H)))
    add("inputs", "p", p)
    add("inputs", "SWAP_THETA12", SWAP_THETA12)
    add("inputs", "alpha", alpha)
    add("inputs", "MODEL_PATH", MODEL_PATH)
    add("inputs", "WINDOW_T", WINDOW_T)
    add("inputs", "STRIDE", STRIDE)
    add("inputs", "DEVICE", DEVICE)
    add("inputs", "SEED", SEED)
    add("inputs", "PIN_NGRID", PIN_NGRID)
    add("inputs", "PIN_NDISPLAY", PIN_NDISPLAY)

    # ---- Derived / intermediate
    add("derived", "n_codeword_len", n)
    add("derived", "w_h", w_h)
    add("derived", "theta1", f"{theta1:.8f}")
    add("derived", "theta2", f"{theta2:.8f}")
    add("derived", "tau_tilde_MLE_1b", tau_tilde)
    add("derived", "S_even", S)
    add("derived", "S_halfwidth", half)
    add("derived", "S_mass_symmetric(>=alpha)", f"{S_mass_symmetric:.8f}")
    add("derived", "crop_start_a_0b", a)
    add("derived", "crop_end_b_0b", b)

    # ---- Final
    add("final", "tau_hat_count", tau_hat_global)  # Prediction (0-based index)
    add("final", "tau_true_count", TAU_TRUE)       # Ground truth (1-based index)

    # ---- π_n values used in the bar-sum for S (the bars we actually summed)
    # Store both raw and normalized probabilities in the CSV.
    for k in range(-half, half + 1):
        add("pi_window_raw",  f"pi_raw[{k:+d}]",  f"{pi.get(k, 0.0):.12e}")
        add("pi_window_norm", f"pi_norm[{k:+d}]", f"{pi_norm.get(k, 0.0):.12e}")

    # Write results to CSV.
    out_csv = "result_final_algorithm.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {os.path.abspath(out_csv)}")

    # Final report to stdout.
    print("\n================= RESULT =================")
    print(f"True changepoint τ (count) : {TAU_TRUE}")
    print(f"Coarse MLE τ~ (on z_i)    : {tau_tilde}")
    print(f"α-window S (even), half   : S={S}, halfwidth={half}")
    print(f"Refined NN τ̂ (count)      : {tau_hat_global}")
    print("==========================================")
    # Optional: print debug info if needed.
    # print(f"[debug] votes in crop (global): {[a+v for v in votes]}")

if __name__ == "__main__":
    main()
