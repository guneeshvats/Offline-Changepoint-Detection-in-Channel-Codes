# =============================================================================
# final_algorithm_unknown_p.py
# =============================================================================
# "Changepoint Detection with Neural Refinement (Unknown-p case)" — Reference
# Implementation.
#
# This script implements the UNKNOWN-p pipeline described in the paper:
#
#   1. Load two CSV files containing pools of binary codewords (C1 and C2).
#   2. Construct a sequence of length M with a true changepoint at tau_true:
#        - first tau_true codewords sampled from C1,
#        - remaining (M - tau_true) sampled from C2.
#   3. Project each codeword onto a user-specified binary vector h to obtain
#      a Bernoulli sequence z_i = <y_i, h> mod 2.
#   4. Since p is UNKNOWN, we do NOT use Bernoulli MLE / Hinkley prior.
#      Instead, we run two classical offline CPD methods on z:
#         - Binary Segmentation (BinSeg)
#         - PELT
#      Each provides a coarse changepoint estimate τ_B, τ_P.
#   5. Around each estimate we take a symmetric window of halfwidth w':
#         S_B  = [τ_B - w', τ_B + w']
#         S_P  = [τ_P - w', τ_P + w']
#      and define the coarse region S as their union:
#         S = S_B ∪ S_P .
#   6. We crop the ORIGINAL codeword sequence y_1^M to this union region and
#      run a TorchScript Type-II CNN (trained across multiple p) using sliding
#      windows of length T.
#   7. We aggregate the CNN scores over candidate τ in S to obtain the final
#      refined estimate τ̂.
#
# This script is therefore the "Unknown-p + Neural Refinement" pipeline.
#
# Requirements:
#   pip install numpy pandas torch ruptures
#
# Notes:
#   - The neural network model must be a TorchScript .pt file that accepts
#     input of shape (B, T, n) and outputs logits of shape (B, T+1).
#   - We use ruptures (BinSeg, PELT) with an L2 cost on the 1D Bernoulli
#     projections z_i, which matches the unknown-p experiments in the paper.
# =============================================================================

from __future__ import annotations
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import ruptures as rpt  # for BinSeg and PELT

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
    c1_pool: np.ndarray,
    c2_pool: np.ndarray,
    M: int,
    tau_true: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct a sequence of codewords of length M with a changepoint at tau_true.
    The first tau_true codewords are sampled (with replacement) from c1_pool,
    and the remaining (M - tau_true) from c2_pool.

    Args:
        c1_pool (np.ndarray): Pool of codewords for segment 1, shape (N1, n).
        c2_pool (np.ndarray): Pool of codewords for segment 2, shape (N2, n).
        M (int): Total sequence length.
        tau_true (int): Index of the changepoint (1-based, must be in [1, M-1]).
        rng (np.random.Generator): Numpy random generator.

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
# Projection with h
# =============================================================================

def project_with_h(y_seq: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Project each codeword y_i in y_seq onto the binary vector h using inner
    product mod 2. This produces the Bernoulli sequence z_i used in BinSeg/PELT.

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

    sums = (y_seq.astype(np.uint8) & h).sum(axis=1)
    return (sums % 2).astype(np.uint8)

# =============================================================================
# Coarse CPD: BinSeg + PELT on z (Unknown-p)
# =============================================================================

def binseg_single_cp(z: np.ndarray) -> int:
    """
    Run Binary Segmentation (BinSeg) with an L2 cost on the 1D sequence z,
    and return a single changepoint estimate τ_B (1-based).

    Args:
        z (np.ndarray): Bernoulli sequence (0/1), shape (M,).

    Returns:
        int: τ_B in {1,...,M-1}.
    """
    M = len(z)
    signal = z.astype(float).reshape(-1, 1)
    algo = rpt.Binseg(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=1)   # returns [τ_B, M]
    if len(bkps) < 1:
        return M // 2
    tau_B = int(bkps[0])
    tau_B = max(1, min(M - 1, tau_B))
    return tau_B


def pelt_single_cp(z: np.ndarray, pen: float) -> int:
    """
    Run PELT with an L2 cost on z and return a single changepoint estimate τ_P (1-based).

    Args:
        z (np.ndarray): Bernoulli sequence (0/1), shape (M,).
        pen (float): Penalty parameter for PELT (controls sensitivity).

    Returns:
        int: τ_P in {1,...,M-1}.
    """
    M = len(z)
    signal = z.astype(float).reshape(-1, 1)
    algo = rpt.Pelt(model="l2").fit(signal)
    bkps = algo.predict(pen=pen)   # usually [τ_P, M]
    if len(bkps) == 0:
        return M // 2
    # Filter out the final M if present
    candidates = [b for b in bkps if b < M]
    if not candidates:
        return M // 2
    tau_P = int(candidates[0])
    tau_P = max(1, min(M - 1, tau_P))
    return tau_P


def coarse_window_from_binseg_pelt(
    z: np.ndarray,
    halfwidth: int,
    pelt_penalty: float,
    M: int,
) -> Tuple[int, int, int, int, Dict[str, Tuple[int, int]]]:
    """
    Given the Bernoulli projection z, perform BinSeg and PELT to obtain coarse
    estimates τ_B and τ_P, then build symmetric windows around each and return
    their union as the coarse refinement region S.

    Args:
        z (np.ndarray): Bernoulli sequence (0/1), shape (M,).
        halfwidth (int): Halfwidth w' for each coarse window.
        pelt_penalty (float): Penalty for PELT.

        M (int): Total sequence length (for clamping).

    Returns:
        (tau_B_1b, tau_P_1b, a_union, b_union, windows_dict)
          tau_B_1b: BinSeg estimate (1-based)
          tau_P_1b: PELT estimate (1-based)
          a_union, b_union: 0-based inclusive indices of S = S_B ∪ S_P
          windows_dict: dict with entries "SB"=(aB,bB), "SP"=(aP,bP)
    """
    tau_B_1b = binseg_single_cp(z)
    tau_P_1b = pelt_single_cp(z, pen=pelt_penalty)

    centerB0 = tau_B_1b - 1
    centerP0 = tau_P_1b - 1

    aB, bB = crop_indices(centerB0, halfwidth, M)
    aP, bP = crop_indices(centerP0, halfwidth, M)

    a_union = min(aB, aP)
    b_union = max(bB, bP)

    windows = {"SB": (aB, bB), "SP": (aP, bP)}
    return tau_B_1b, tau_P_1b, a_union, b_union, windows

# =============================================================================
# Neural Network Refinement on Raw Codewords
# =============================================================================

def nn_refine_tau_soft(
    y_crop: np.ndarray,
    window_T: int,
    model_path: str,
    a: int,
    b: int,                   # crop bounds (0-based, inclusive)
    stride: int = 1,
    device: str = "cpu",
) -> Tuple[int, int, np.ndarray]:
    """
    Refine the changepoint estimate using a neural network model (Type-II).
    Slides a window of length T over the cropped codeword sequence and
    aggregates neural network probabilities for candidate τ.

    Args:
        y_crop (np.ndarray): Cropped codeword sequence, shape (L, n).
        window_T (int): Window length for the neural network.
        model_path (str): Path to the TorchScript model (.pt file).
        a (int): Start index of the crop (0-based, inclusive).
        b (int): End index of the crop (0-based, inclusive).
        stride (int): Step size for sliding window.
        device (str): Device for model inference ("cpu" or "cuda").

    Returns:
        (tau_hat_crop_0b, tau_hat_global_0b, scores)
          tau_hat_crop_0b: Index of τ̂ inside the crop (0-based).
          tau_hat_global_0b: Index of τ̂ in the global sequence (0-based).
          scores: Aggregated scores for each position in the crop.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for NN refinement (torch not available).")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    L, n = y_crop.shape
    if L < window_T:
        # Minimal safeguard: we rely on the caller to ensure a sufficiently
        # large crop, but do not hard-fail here.
        raise ValueError(f"Crop length ({L}) must be >= window_T ({window_T}).")

    # Load TorchScript model for inference.
    jit_model = torch.jit.load(model_path, map_location=device)
    jit_model.to(device)
    jit_model.eval()

    scores = np.zeros(L, dtype=np.float64)  # score per candidate tau in the crop

    with torch.no_grad():
        s = 0
        while s + window_T <= L:
            window = y_crop[s: s + window_T]  # (T, n)
            x = torch.from_numpy(window[None, ...].astype(np.float32)).to(device)
            logits = jit_model(x)  # (1, T+1)

            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # p(t), t=0..T
            mass_interior = float(probs[1:window_T].sum())
            if mass_interior < 1e-8:
                s += stride
                continue

            for t in range(1, window_T):  # interior positions
                tau_global_0b = a + s + t
                if a <= tau_global_0b <= b:
                    idx = tau_global_0b - a
                    scores[idx] += float(probs[t])
            s += stride

    if np.allclose(scores, 0.0):
        # Fallback: if the network gives no useful mass, take the center.
        tau_hat_global_0b = (a + b) // 2
        tau_hat_crop_0b = tau_hat_global_0b - a
        return tau_hat_crop_0b, tau_hat_global_0b, scores

    tau_hat_crop_0b = int(np.argmax(scores))
    tau_hat_global_0b = a + tau_hat_crop_0b
    return tau_hat_crop_0b, tau_hat_global_0b, scores

# =============================================================================
# Utility Functions
# =============================================================================

def clamp(val: int, lo: int, hi: int) -> int:
    """Clamp an integer value to the range [lo, hi]."""
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
        (a, b): inclusive start/end indices (0-based).
    """
    a = clamp(center - halfwidth, 0, M - 1)
    b = clamp(center + halfwidth, 0, M - 1)
    return a, b

# =============================================================================
# Main Driver Function with User Inputs (Unknown-p)
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # ========================= USER INPUTS ===================================
    # -------------------------------------------------------------------------
    # 1) Paths to CSVs containing codeword pools for C1 and C2.
    #    These CSVs are generated by the MATLAB scripts in Data_Generation/.
    CSV_C1: str = "/path/to/bsc_p0.05_codewords1.csv"
    CSV_C2: str = "/path/to/bsc_p0.05_codewords2.csv"

    # 2) Global sequence length M and TRUE changepoint (for synthetic tests).
    #    For a single sequence experiment set MC_ENABLE = False and pick a
    #    convenient TAU_TRUE in {1,...,M-1}.
    M: int = 50
    TAU_TRUE: int = 20

    # 3) Test vector h (binary length-n).
    #    For BCH n=15 or n=31 or LDPC n=648, obtain h from:
    #      - Find_min_weight_h_BCH_n15_n31.m
    #      - Find_min_weight_h_LDPC_n648.m
    #    h must satisfy h ∈ C2^⊥ \ C1^⊥ and should have small Hamming weight.
    H: List[int] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # example for n=15

    # 4) Coarse CPD window parameters for BinSeg/PELT.
    #    Each coarse estimate (τ_B, τ_P) gets a symmetric window of halfwidth
    #    COARSE_HALFWIDTH around it; we then take the union of these windows.
    COARSE_HALFWIDTH: int = 20  # w' in the paper
    PELT_PENALTY: float = 5.0   # penalty parameter for PELT (tune per dataset)

    # 5) Neural network model (Type-II CNN) configuration.
    #    MODEL_PATH is a TorchScript .pt file produced from the notebooks in
    #    Deep_Learning_Approach/BCH+BSC_unknown_p/.
    MODEL_PATH: str = "/path/to/n15_unknown_p_T5_model.pt"
    WINDOW_T: int = 5
    STRIDE: int = 1
    DEVICE: str = "cpu"

    # 6) Random seed (for sequence construction and Monte Carlo runs).
    SEED: int = 12345

    # 7) Monte Carlo controls.
    #    If MC_ENABLE = True, we draw tau_true uniformly from {1,...,M-1} in
    #    each trial and report exact-hit statistics.
    MC_ENABLE: bool = False
    MC_NR: int = 1000

    # -------------------------------------------------------------------------
    # ======================= END USER INPUTS =================================
    # -------------------------------------------------------------------------

    # Load codeword pools.
    c1 = load_codewords_csv(CSV_C1)
    c2 = load_codewords_csv(CSV_C2)
    n = c1.shape[1]
    if c2.shape[1] != n:
        raise ValueError("CSV pools have different codeword lengths.")

    # Validate h.
    h = np.array(H, dtype=np.uint8)
    if h.shape != (n,):
        raise ValueError(f"h must have length n={n}, but got {len(h)}.")
    w_h = int(h.sum())
    print(f"[info] n={n}, w(h)={w_h}")

    rng = np.random.default_rng(SEED)

    # =========================================================================
    # Monte Carlo Mode (optional)
    # =========================================================================
    if MC_ENABLE:
        print(f"\n[MC] Running {MC_NR} trials with random true τ ∈ {{1,…,{M-1}}}")
        exact = 0
        mc_rows = []

        for r in range(MC_NR):
            tau_true = int(rng.integers(1, M))  # τ_true ∈ {1,...,M-1}
            y_seq = build_sequence_from_pools(c1, c2, M, tau_true, rng)
            z = project_with_h(y_seq, h)

            tau_B, tau_P, a_union, b_union, windows = coarse_window_from_binseg_pelt(
                z=z,
                halfwidth=COARSE_HALFWIDTH,
                pelt_penalty=PELT_PENALTY,
                M=M,
            )

            y_crop = y_seq[a_union: b_union + 1]
            if len(y_crop) < WINDOW_T:
                need = WINDOW_T - len(y_crop)
                a2 = clamp(a_union - (need // 2), 0, M - 1)
                b2 = clamp(b_union + (need - (a_union - a2)), 0, M - 1)
                y_crop = y_seq[a2: b2 + 1]
                a_union, b_union = a2, b2

            _, tau_hat_global_0b, _ = nn_refine_tau_soft(
                y_crop=y_crop,
                window_T=WINDOW_T,
                model_path=MODEL_PATH,
                a=a_union,
                b=b_union,
                stride=STRIDE,
                device=DEVICE,
            )
            tau_hat_1b = tau_hat_global_0b + 1  # convert to 1-based

            exact += int(tau_hat_1b == tau_true)
            mc_rows.append(
                {
                    "run": r,
                    "tau_true_1b": tau_true,
                    "tau_B_binseg_1b": tau_B,
                    "tau_P_pelt_1b": tau_P,
                    "S_union_a_0b": a_union,
                    "S_union_b_0b": b_union,
                    "tau_hat_1b": tau_hat_1b,
                }
            )

        acc = exact / MC_NR
        print(f"\n[MC] exact hits = {exact}/{MC_NR} ({100*acc:.2f}%)")
        mc_csv = "mc_runs_unknown_p.csv"
        pd.DataFrame(mc_rows).to_csv(mc_csv, index=False)
        print(f"[MC] saved per-run results -> {os.path.abspath(mc_csv)}")
        return

    # =========================================================================
    # Single-Run Mode (one sequence with specified TAU_TRUE)
    # =========================================================================
    y_seq = build_sequence_from_pools(c1, c2, M, TAU_TRUE, rng)
    z = project_with_h(y_seq, h)

    tau_B, tau_P, a_union, b_union, windows = coarse_window_from_binseg_pelt(
        z=z,
        halfwidth=COARSE_HALFWIDTH,
        pelt_penalty=PELT_PENALTY,
        M=M,
    )

    print(f"[coarse-BinSeg] τ_B = {tau_B}   (true τ = {TAU_TRUE})")
    print(f"[coarse-PELT ] τ_P = {tau_P}   (true τ = {TAU_TRUE})")
    print(f"[coarse-union] S = [{a_union}:{b_union}] (0-based, inclusive)")

    y_crop = y_seq[a_union: b_union + 1]
    if len(y_crop) < WINDOW_T:
        print("[warn] Crop shorter than WINDOW_T; expanding window.")
        need = WINDOW_T - len(y_crop)
        a2 = clamp(a_union - (need // 2), 0, M - 1)
        b2 = clamp(b_union + (need - (a_union - a2)), 0, M - 1)
        y_crop = y_seq[a2: b2 + 1]
        a_union, b_union = a2, b2
        print(f"[crop-adjusted] S' = [{a_union}:{b_union}] (len={len(y_crop)})")

    _, tau_hat_global_0b, scores = nn_refine_tau_soft(
        y_crop=y_crop,
        window_T=WINDOW_T,
        model_path=MODEL_PATH,
        a=a_union,
        b=b_union,
        stride=STRIDE,
        device=DEVICE,
    )
    tau_hat_1b = tau_hat_global_0b + 1

    # Save results to CSV for analysis.
    rows = []
    def add(section, name, value):
        rows.append({"section": section, "name": name, "value": value})

    # Inputs
    add("inputs", "CSV_C1", CSV_C1)
    add("inputs", "CSV_C2", CSV_C2)
    add("inputs", "M", M)
    add("inputs", "TAU_TRUE_1b", TAU_TRUE)
    add("inputs", "H_vector", " ".join(map(str, H)))
    add("inputs", "COARSE_HALFWIDTH", COARSE_HALFWIDTH)
    add("inputs", "PELT_PENALTY", PELT_PENALTY)
    add("inputs", "MODEL_PATH", MODEL_PATH)
    add("inputs", "WINDOW_T", WINDOW_T)
    add("inputs", "STRIDE", STRIDE)
    add("inputs", "DEVICE", DEVICE)
    add("inputs", "SEED", SEED)

    # Derived / coarse
    add("derived", "n_codeword_len", n)
    add("derived", "w_h", w_h)
    add("derived", "tau_B_binseg_1b", tau_B)
    add("derived", "tau_P_pelt_1b", tau_P)
    add("derived", "S_union_a_0b", a_union)
    add("derived", "S_union_b_0b", b_union)

    # Final
    add("final", "tau_hat_1b", tau_hat_1b)
    add("final", "tau_true_1b", TAU_TRUE)

    out_csv = "result_final_algorithm_unknown_p.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {os.path.abspath(out_csv)}")

    print("\n================= RESULT (Unknown-p) =================")
    print(f"True changepoint τ_true        : {TAU_TRUE}")
    print(f"BinSeg coarse τ_B             : {tau_B}")
    print(f"PELT  coarse τ_P              : {tau_P}")
    print(f"Union coarse region S (0-based): [{a_union}:{b_union}]")
    print(f"Refined NN τ̂ (1-based)        : {tau_hat_1b}")
    print("=====================================================")


if __name__ == "__main__":
    main()
