# binomial_changepoint_sec2_2.py
# ---------------------------------------------------------------------
# Hinkley & Hinkley (1970): up to the end of §2.2 for Bernoulli data.
# Implements:
#   • Simulate R_i (Eq. 1.1)
#   • X_t (Eq. 2.2) and MLE tau_hat = argmax X_t
#   • Random walks W, W' from X_t differences (Eq. 2.6 → 2.7)
#   • Maxima (M, I) and (M', I'), and tau_hat via (Eq. 2.13)
#   • p00 and p00' (Eqs. 2.18 & 2.21)
#   • Recursions for q_{l,m}, p_{l,m} and q'_{l,m}, p'_{l,m} (Eqs. 2.15, 2.17, 2.19, 2.20)
#   • Plots and CSVs for p_{l,m} and p'_{l,m}
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# Helpers

def _log_binom_pmf(n, r, p):
    # log [ C(n,r) p^r (1-p)^(n-r) ]
    from math import lgamma, log
    return (lgamma(n+1) - lgamma(r+1) - lgamma(n-r+1)
            + r*log(p) + (n-r)*log(1.0-p))

def _logsumexp(log_vals):
    # stable sum(exp(.))
    m = max(log_vals)
    if m == -float('inf'):
        return m
    s = sum(math.exp(v - m) for v in log_vals)
    return m + math.log(s)




# ====================== Basics ======================

def simulate_bernoulli_sequence(T, tau, theta0, theta1, seed=123):
    """
    Generate R_1..R_T per Eq. (1.1) using Bernoulli draws:
        P(R_i=1)=theta0 for i<=tau, and P(R_i=1)=theta1 for i>tau.
    """
    # Basic edge case checks for tau & Bernoulli Parameters
    if not (1 <= tau < T):
        raise ValueError("tau must be in {1,...,T-1}.")
    if not (0 < theta0 < 1 and 0 < theta1 < 1):
        raise ValueError("theta0 and theta1 must be in (0,1).")

    rng = np.random.default_rng(seed)
    R = np.zeros(T, dtype=int)
    R[:tau] = rng.binomial(1, theta0, size=tau)  # Bernoulli(theta0)
    R[tau:] = rng.binomial(1, theta1, size=T - tau)  # Bernoulli(theta1)
    return R


# ====================== Eq. (2.2) ======================

def compute_Xt(R, theta0, theta1):
    """
    Contrast sequence X_t (Eq. 2.2), for t=1..T-1:
        X_t = sum_{i=1}^t [ R_i log(theta0/theta1) + (1-R_i) log((1-theta0)/(1-theta1)) ].
    """
    a = math.log(theta0 / theta1)
    b = math.log((1 - theta0) / (1 - theta1))
    increments = R * a + (1 - R) * b
    csum = np.cumsum(increments)
    return csum[:-1]  # t = 1..T-1


def tau_hat_from_Xt(Xt, tie_rule="smallest"):
    """
    MLE tau_hat = argmax_t X_t with tie-breaking:
      - 'smallest' (paper's rule (i)) or 'largest' (rule (ii)).
    Xt[k-1] = X_k
    """
    mx = float(np.max(Xt))
    idxs = np.flatnonzero(np.isclose(Xt, mx))
    return int((idxs[0] if tie_rule == "smallest" else idxs[-1]) + 1)


# ====================== Eq. (2.5), (2.6), (2.7), (2.13) ======================

def phi(theta0, theta1):
    """
    Phi (Eq. 2.5):
        Φ = log((1-θ1)/(1-θ0)) / log(θ0/θ1)
    """
    return math.log((1 - theta1) / (1 - theta0)) / math.log(theta0 / theta1)


def build_walks_from_X(Xt, tau, theta0, theta1):
    """
    Build random-walk increments from X_t differences (Eq. 2.6):
      Left side (i=1..tau-1):   Y_i  = (X_{tau-i} - X_{tau-i+1}) / log(θ0/θ1)
      Right side (i=1..T-tau-1):Y'_i = (X_{tau+i} - X_{tau+i-1}) / log(θ0/θ1)
    Then cumulative sums to get paths W and W' (Eq. 2.7).
    Returns dicts for left and right: {Y, path, S, M, I}.
    """
    a = math.log(theta0 / theta1)
    Tm1 = len(Xt)  # == T-1

    # Left increments
    left_incs = []
    for i in range(1, tau):           # i=1..tau-1
        t = tau - i                   # t in {1..T-1}
        x_t = Xt[t - 1]
        x_tp1 = Xt[t] if t < Tm1 else Xt[-1]
        left_incs.append((x_t - x_tp1) / a)
    Y_left = np.array(left_incs, dtype=float)
    path_left = np.concatenate([[0.0], np.cumsum(Y_left)])
    S_left = float(np.max(path_left[1:])) if len(path_left) > 1 else 0.0
    if S_left > 0:
        I_left = int(np.argmax(path_left[1:]) + 1)
        M_left = S_left
    else:
        I_left = 0
        M_left = 0.0

    # Right increments
    right_incs = []
    for i in range(1, (Tm1 - tau) + 1):  # count = T - tau - 1
        t = tau + i
        x_t = Xt[t - 1]
        x_tm1 = Xt[t - 2]
        right_incs.append((x_t - x_tm1) / a)
    Y_right = np.array(right_incs, dtype=float)
    path_right = np.concatenate([[0.0], np.cumsum(Y_right)])
    S_right = float(np.max(path_right[1:])) if len(path_right) > 1 else 0.0
    if S_right > 0:
        I_right = int(np.argmax(path_right[1:]) + 1)
        M_right = S_right
    else:
        I_right = 0
        M_right = 0.0

    left = dict(Y=Y_left, path=path_left, S=S_left, M=M_left, I=I_left)
    right = dict(Y=Y_right, path=path_right, S=S_right, M=M_right, I=I_right)
    return left, right


def tau_hat_from_walks(tau, left, right):
    """
    Combine maxima using Eq. (2.13) to get tau_hat.
    Ties broken to the smallest t (move left).
    """
    if right["M"] > left["M"]:
        return tau + right["I"]
    if left["M"] > right["M"]:
        return tau - left["I"]
    if left["M"] == 0 and right["M"] == 0:
        return tau
    return tau - left["I"]


# ====================== §2.2 constants: p00 and p00' ======================

def p00_left(theta0, Phi, nmax=2000, tol=1e-12):
    """
    p00 = exp( - sum_{n>=1} (1/n) * P(S_n > 0) ),  where
    P(S_n>0) = sum_{r=0}^{floor(n*Phi/(1+Phi))} Binom(n, r; theta0).
    Computed stably in log-space.
    """
    acc = 0.0
    for n in range(1, nmax+1):
        r_max = int(math.floor(n * Phi / (1 + Phi)))
        if r_max < 0:
            inner = 0.0
        else:
            logs = [_log_binom_pmf(n, r, theta0) for r in range(0, r_max+1)]
            logsum = _logsumexp(logs)
            inner = math.exp(logsum)  # <= 1, safe to exp
        inc = inner / n
        acc += inc
        if inc < tol:
            break
    return math.exp(-acc)

def p00_right(theta1, Phi, nmax=2000, tol=1e-12):
    """
    p00' = exp( - sum_{n>=1} (1/n) * P(S'_n > 0) ),  where
    P(S'_n>0) = sum_{r=r_min..n} Binom(n, r; theta1),
    r_min = floor(n*Phi/(1+Phi)) + 1.
    """
    acc = 0.0
    for n in range(1, nmax+1):
        r_min = int(math.floor(n * Phi / (1 + Phi))) + 1
        if r_min > n:
            tail = 0.0
        else:
            logs = [_log_binom_pmf(n, r, theta1) for r in range(r_min, n+1)]
            logsum = _logsumexp(logs)
            tail = math.exp(logsum)
        inc = tail / n
        acc += inc
        if inc < tol:
            break
    return math.exp(-acc)



# ====================== §2.2 recursions for q and p ======================

def q_p_left(theta0, Phi, N, p00):
    """
    Compute q_{l,m} and p_{l,m} on grid l+m<=N for the left walk (Eqs. 2.15 & 2.17).
    Returns dictionaries {(l,m): value}.
    """
    q = {}

    # Boundaries (Eq. 2.17):
    q[(1, 0)] = theta0 * p00
    for l in range(2, N + 1):
        q[(l, 0)] = 0.0
    for m in range(1, N + 1):
        q[(0, m)] = ((1 - theta0) ** m) * p00

    # Interior by increasing l+m (needs neighbors)
    for s in range(2, N + 1):
        for l in range(1, s):
            m = s - l
            val = -l + Phi * m
            if val > Phi:
                q[(l, m)] = theta0 * q.get((l - 1, m), 0.0) + (1 - theta0) * q.get((l, m - 1), 0.0)
            elif (-1 < val < Phi):
                q[(l, m)] = theta0 * q.get((l - 1, m), 0.0)
            else:
                q[(l, m)] = 0.0

    # p_{l,m} from q_{l,m} (Eq. 2.15)
    p = {(l, m): (v if (-l + Phi * m) >= 0 else 0.0) for (l, m), v in q.items()}
    return q, p


def q_p_right(theta1, Phi, N, p00p):
    """
    Compute q'_{l,m} and p'_{l,m} on grid l+m<=N for the right walk (Eqs. 2.19–2.21 / 2.20).
    Returns dictionaries {(l,m): value}.
    """
    q = {}

    # Boundaries (Eq. 2.20):
    q[(0, 1)] = (1 - theta1) * p00p
    for n in range(2, N + 1):
        q[(0, n)] = 0.0
    for l in range(1, N + 1):
        q[(l, 0)] = (theta1 ** l) * p00p

    # Interior
    for s in range(2, N + 1):
        for l in range(1, s):
            m = s - l
            val = l - Phi * m
            if val > 1:
                q[(l, m)] = theta1 * q.get((l - 1, m), 0.0) + (1 - theta1) * q.get((l, m - 1), 0.0)
            elif (-Phi < val < 1):
                q[(l, m)] = theta1 * q.get((l - 1, m), 0.0)
            else:
                q[(l, m)] = 0.0

    # p'_{l,m} from q'_{l,m} (Eq. 2.19)
    p = {(l, m): (v if (l - Phi * m) >= 0 else 0.0) for (l, m), v in q.items()}
    return q, p


# ====================== Plot helpers & CSV ======================

def annotate_points(x, y):
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=8)

def plot_Xt(Xt, tau, tau_hat, outdir):
    tgrid = np.arange(1, len(Xt) + 1)
    plt.figure(figsize=(9, 4.5))
    plt.plot(tgrid, Xt, marker='o')
    plt.axvline(tau, linestyle='--', label=f"true τ = {tau}")
    plt.axvline(tau_hat, linestyle='--', label=f"τ̂ = {tau_hat}")
    plt.title("Contrast $X_t$ (Eq. 2.2)")
    plt.xlabel("t")
    plt.ylabel("$X_t$")
    annotate_points(tgrid, Xt)
    plt.legend(); plt.tight_layout()
    path = os.path.join(outdir, "Xt_contrast.png")
    plt.savefig(path, dpi=160); plt.show()
    print(f"[saved] {path}")

def plot_walk(path_vals, M, I, title, fname, outdir):
    steps = np.arange(len(path_vals))
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, path_vals, marker='o')
    plt.axhline(0, linewidth=1)
    if M > 0 and I > 0:
        plt.scatter([I], [M])
        plt.annotate(f"M={M:.2f}\nI={I}", (I, M), textcoords="offset points", xytext=(6, 6), fontsize=9)
    annotate_points(steps, path_vals)
    plt.title(title); plt.xlabel("steps"); plt.ylabel("cumulative sum")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=160); plt.show()
    print(f"[saved] {path}")

def pmf_dict_to_df(pdict, Phi, left_side=True):
    rows = []
    for (l, m), prob in pdict.items():
        value = (-l + Phi * m) if left_side else (l - Phi * m)
        rows.append((l, m, prob, value))
    df = pd.DataFrame(rows, columns=["l", "m", "prob", "value"])
    return df.sort_values(["l", "m"]).reset_index(drop=True)

def save_df_csv(df, path):
    df.to_csv(path, index=False)
    print(f"[saved] {path}")

def plot_heatmap_from_df(df, title, outpath):
    Lmax = int(df["l"].max()); Mmax = int(df["m"].max())
    G = np.zeros((Lmax + 1, Mmax + 1))
    for _, r in df.iterrows():
        G[int(r.l), int(r.m)] = float(r.prob)
    plt.figure(figsize=(7.2, 5.2))
    im = plt.imshow(G, origin='lower', aspect='auto')
    plt.colorbar(im, label="probability mass")
    plt.xlabel("m"); plt.ylabel("l"); plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160); plt.show()
    print(f"[saved] {outpath}")

def plot_detection_comparison(Xt, tau_true, tau_hat_xt, tau_hat_walk, outdir):
    """Compare changepoint estimates from Eq. (2.2) vs. random-walk Eq. (2.13)."""

    t = np.arange(1, len(Xt)+1)

    plt.figure(figsize=(10, 4.8))
    # Contrast curve
    plt.plot(t, Xt, marker='o', label=r'$X_t$ (Eq. 2.2)')

    # Ground truth and both estimates
    plt.axvline(tau_true, color='k', linestyle='--', label=f'true $\\tau$ = {tau_true}')
    plt.axvline(tau_hat_xt, color='C3', linestyle='-',  label=f'$\\hat\\tau$ from $X_t$ = {tau_hat_xt}')
    plt.axvline(tau_hat_walk, color='C2', linestyle='-.', label=f'$\\hat\\tau$ from walks = {tau_hat_walk}')

    # If they disagree, shade the gap
    if tau_hat_xt != tau_hat_walk:
        lo, hi = sorted([tau_hat_xt, tau_hat_walk])
        ymin = min(Xt) - 0.05*abs(min(Xt))
        ymax = max(Xt) + 0.05*abs(max(Xt))
        plt.fill_betweenx([ymin, ymax], lo, hi, color='orange', alpha=0.15, label='disagreement')

    plt.xlabel('t'); plt.ylabel(r'$X_t$')
    plt.title('Change-point estimates: MLE (Eq. 2.2) vs Random-walk (Eq. 2.13)')
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(outdir, 'compare_tauhat_xt_vs_walk.png')
    plt.savefig(fn, dpi=160)
    plt.show()
    print(f'[saved] {fn}')


# ====================== Main ======================

def main():
    # ----------- change inputs here -----------
    T       = 120
    tau     = 70
    theta0  = 0.80   # theta0 > theta1 [Assumption by the paper to keep phi as positive]
    theta1  = 0.50
    seed    = 42   #27   #42
    Ngrid   = 25          # compute q,p on grid l+m <= Ngrid
    tieRule = "smallest"  # 'smallest' or 'largest' max X_t
    outdir  = "outputs_sec2_2"
    # ------------------------------------------

    os.makedirs(outdir, exist_ok=True)
    print("=== Inputs ===")
    print(f"T={T}, tau={tau}, theta0={theta0}, theta1={theta1}, seed={seed}, Ngrid={Ngrid}, tieRule={tieRule}")

    # 1) Simulate Bernoulli sequence (Eq. 1.1)
    R = simulate_bernoulli_sequence(T, tau, theta0, theta1, seed)

    # 2) Compute X_t and MLE tau_hat (Eq. 2.2)
    Xt = compute_Xt(R, theta0, theta1)
    tau_hat = tau_hat_from_Xt(Xt, tie_rule=tieRule)
    plot_Xt(Xt, tau, tau_hat, outdir)

    # 3) Build walks from X_t differences (Eq. 2.6–2.7) and tau_hat via (2.13)
    # left, right = build_walks_from_X(Xt, tau, theta0, theta1)
    # tau_hat_walk = tau_hat_from_walks(tau, left, right)
    t_star = tau_hat_from_Xt(Xt, tie_rule=tieRule)            # center at MLE argmax
    left, right = build_walks_from_X(Xt, t_star, theta0, theta1)
    tau_hat_walk = tau_hat_from_walks(t_star, left, right)

    # Comparison from MLE and Random Walk CPD 
    plot_detection_comparison(Xt, tau, tau_hat, tau_hat_walk, outdir)

    print("\n=== Estimation summary ===")
    print(f"true τ               : {tau}")
    print(f"τ̂ from X_t          : {tau_hat}")
    print(f"τ̂ from walks (2.13)  : {tau_hat_walk}")
    print(f"match? {'YES' if tau_hat == tau_hat_walk else 'NO'}")

    plot_walk(left["path"], left["M"], left["I"],
              "Left walk W (from X_{τ-i} - X_{τ-i+1})", "walk_left.png", outdir)
    plot_walk(right["path"], right["M"], right["I"],
              "Right walk W' (from X_{τ+i} - X_{τ+i-1})", "walk_right.png", outdir)

    # 4) §2.2 constants
    Phi = phi(theta0, theta1)
    p00  = p00_left(theta0, Phi)
    p00p = p00_right(theta1, Phi)

    print("\n=== §2.2 constants ===")
    print(f"Φ   = {Phi:.6f}")
    print(f"p00  (left never >0)  = {p00:.6f}   (Eq. 2.18)")
    print(f"p00' (right never >0) = {p00p:.6f}  (Eq. 2.21)")

    # 5) Recursions for q,p on both sides (Eqs. 2.15, 2.17, 2.19, 2.20)
    q_left,  p_left  = q_p_left(theta0, Phi, Ngrid, p00)
    q_right, p_right = q_p_right(theta1, Phi, Ngrid, p00p)

    # 6) Save PMFs and show heatmaps
    df_p_left  = pmf_dict_to_df(p_left,  Phi, left_side=True)
    df_p_right = pmf_dict_to_df(p_right, Phi, left_side=False)

    save_df_csv(df_p_left,  os.path.join(outdir, "p_left_maxima.csv"))
    save_df_csv(df_p_right, os.path.join(outdir, "p_right_maxima.csv"))

    plot_heatmap_from_df(df_p_left,  "Heatmap p_{l,m} (left maxima PMF)",  os.path.join(outdir, "heatmap_p_left.png"))
    plot_heatmap_from_df(df_p_right, "Heatmap p'_{l,m} (right maxima PMF)", os.path.join(outdir, "heatmap_p_right.png"))

    print("\n=== Mass on grid (excluding atom at 0) ===")
    print(f"Σ p_left  = {df_p_left['prob'].sum():.6f};  add atom p00  = {p00:.6f}")
    print(f"Σ p_right = {df_p_right['prob'].sum():.6f}; add atom p00' = {p00p:.6f}")
    print(f"\nAll outputs saved in: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
