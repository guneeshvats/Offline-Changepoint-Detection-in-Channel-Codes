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
#   • Optionally, save all results and plots in a zip file
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import zipfile

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




# ====== Section 3: Asymptotic distribution pi_n (Eq. 3.1) ======

def _sorted_cdf_arrays_from_pdict(pdict, phi, which):
    """
    which = 'left'  -> order by  (-l + phi*m)   (for p_{jk} inner sums in pi_{+n})
          = 'right' -> order by  ( l - phi*m)   (for p'_{jk} inner sums in pi_{-n})
    Returns (vals_sorted, cums_sorted), both numpy arrays.
    """
    import numpy as np
    vals, probs = [], []
    for (l, m), pr in pdict.items():
        if pr <= 0.0:
            continue
        if which == 'left':
            v = -l + phi * m
        else:  # 'right'
            v = l - phi * m
        vals.append(float(v))
        probs.append(float(pr))
    if not vals:
        return np.array([0.0]), np.array([0.0])
    vals = np.asarray(vals)
    probs = np.asarray(probs)
    order = np.argsort(vals)
    vals = vals[order]
    cums = np.cumsum(probs[order])
    return vals, cums

def _cdf_less_than(thresh, vals_sorted, cums_sorted, atom_prob=0.0, atom_at_zero=True):
    """
    Returns sum of masses with value < thresh, plus atom at 0 if thresh>0.
    """
    idx = np.searchsorted(vals_sorted, thresh, side='left')
    mass = float(cums_sorted[idx-1]) if idx > 0 else 0.0
    if atom_at_zero and thresh > 0:
        mass += float(atom_prob)
    return mass

def compute_pi_distribution(phi, p_left, p_right, p00, p00p, n_max):
    """
    Implements Eq. (3.1):
      pi_0 = p00 * p00p
      pi_{-n} = sum_{l=0}^n p_{l, n-l} * sum_{j,k >=0, j-phi*k < -l + phi(n-l)} p'_{jk}
      pi_{+n} = sum_{l=0}^n p'_{l, n-l} * sum_{j,k >=0, -j+phi*k <  l - phi(n-l)} p_{jk}
    """
    # Precompute fast CDFs for the inner infinite sums
    # p: order by (-j + phi*k) ; p': order by (j - phi*k)
    vals_p, cums_p = _sorted_cdf_arrays_from_pdict(p_left,  phi, which='left')
    vals_pp,cums_pp= _sorted_cdf_arrays_from_pdict(p_right, phi, which='right')

    pi = {0: float(p00 * p00p)}

    # helper to get dictionary value safely
    def _get(d, l, m):
        return float(d.get((l, m), 0.0))

    for n in range(1, n_max+1):
        # pi_{-n}
        s_neg = 0.0
        for l in range(0, n+1):
            m = n - l
            outer = _get(p_left, l, m)
            if outer == 0.0:
                continue
            thresh = -l + phi * m
            inner = _cdf_less_than(thresh, vals_pp, cums_pp, atom_prob=p00p, atom_at_zero=True)
            s_neg += outer * inner
        pi[-n] = s_neg

        # pi_{+n}
        s_pos = 0.0
        for l in range(0, n+1):
            m = n - l
            outer = _get(p_right, l, m)
            if outer == 0.0:
                continue
            thresh =  l - phi * m
            inner = _cdf_less_than(thresh, vals_p, cums_p, atom_prob=p00, atom_at_zero=True)
            s_pos += outer * inner
        pi[+n] = s_pos

    return pi

def choose_Ngrid_auto(theta0, theta1, phi, target_mass=0.99999, start=80, step=40, maxN=1600):
    """
    Increase N until both sides' total mass (grid + atom) exceeds target_mass.
    """
    N = start
    while True:
        qL, pL = q_p_left(theta0, phi, N, p00_left(theta0, phi))
        qR, pR = q_p_right(theta1, phi, N, p00_right(theta1, phi))
        massL = sum(pL.values()) + p00_left(theta0, phi)
        massR = sum(pR.values()) + p00_right(theta1, phi)
        if massL >= target_mass and massR >= target_mass:
            return N, (qL, pL), (qR, pR)
        N += step
        if N > maxN:
            return N, (qL, pL), (qR, pR)

def plot_pi_range(pi_dict, nmax, outpath):
    xs = np.arange(-nmax, nmax+1)
    ys = np.array([pi_dict.get(int(x), 0.0) for x in xs], dtype=float)
    plt.figure(figsize=(10,4.5))
    plt.bar(xs, ys, width=0.8)
    plt.xlabel(r"Relative location $n$ ( $\hat\tau - \tau = n$ )")
    plt.ylabel(r"$\pi_n$")
    plt.title(r"Asymptotic distribution of $\hat\tau-\tau$ (Eq. 3.1)")
    for x,y in zip(xs, ys):
        if y>0:
            plt.annotate(f"{y:.3f}", (x,y), textcoords="offset points", xytext=(0,4), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160); plt.show()
    print(f"[saved] {outpath}")









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

def empirical_pi_distributions(T, theta0, theta1, trials=2000, n_display=10, seed=12345):
    """
    Build empirical histograms of n = (tau_hat - tau) for:
      - walk estimator (Eq. 2.13) with walks centered at TRUE tau
      - MLE estimator (argmax X_t)
    Returns two dicts: emp_walk, emp_mle with keys in [-n_display..n_display]
    """
    rng = np.random.default_rng(seed)
    counts_walk = {n: 0 for n in range(-n_display, n_display+1)}
    counts_mle  = {n: 0 for n in range(-n_display, n_display+1)}

    overflow_walk = 0
    overflow_mle  = 0

    for t in range(trials):
        # random true tau ~ Uniform{1,...,T-1}
        tau_true = int(rng.integers(1, T))
        # jitter seed per trial for independence
        R = simulate_bernoulli_sequence(T, tau_true, theta0, theta1, seed=int(rng.integers(0, 2**31-1)))

        Xt = compute_Xt(R, theta0, theta1)
        # MLE estimator
        tau_hat_mle = tau_hat_from_Xt(Xt, tie_rule="smallest")

        # Walk estimator, IMPORTANT: center at TRUE tau (not MLE)
        left_true, right_true = build_walks_from_X(Xt, tau_true, theta0, theta1)
        tau_hat_walk = tau_hat_from_walks(tau_true, left_true, right_true)

        n_mle  = tau_hat_mle  - tau_true
        n_walk = tau_hat_walk - tau_true

        # clip into display window
        # do NOT clip; send out-of-window mass to 'overflow' counters
        if -n_display <= n_walk <= n_display:
            counts_walk[n_walk] += 1
        else:
            overflow_walk += 1

        if -n_display <= n_mle <= n_display:
            counts_mle[n_mle] += 1
        else:
            overflow_mle += 1

    emp_walk = {n: counts_walk[n] / trials for n in counts_walk}
    emp_mle  = {n: counts_mle[n]  / trials for n in counts_mle}
    tail_walk = overflow_walk / trials
    tail_mle  = overflow_mle  / trials
    return emp_walk, emp_mle, tail_walk, tail_mle



def plot_theory_vs_empirical(pi_theory, emp_walk, emp_mle, n_display, outpath):
    """
    Bar plot for theory pi_n, with markers for empirical walk and MLE histograms.
    """
    xs = np.arange(-n_display, n_display+1)
    y_the = np.array([pi_theory.get(int(x), 0.0) for x in xs], dtype=float)
    y_w   = np.array([emp_walk.get(int(x), 0.0)   for x in xs], dtype=float)
    y_m   = np.array([emp_mle.get(int(x), 0.0)    for x in xs], dtype=float)

    plt.figure(figsize=(11, 5))
    # theory as bars
    plt.bar(xs, y_the, width=0.8, alpha=0.5, label=r"Theory $\pi_n$ (Eq. 3.1)")
    # empirical overlays
    plt.plot(xs, y_w, marker='o', linewidth=1.5, label=r"Empirical (walk, centered at true $\tau$)")
    plt.plot(xs, y_m, marker='x', linewidth=1.5, label=r"Empirical (MLE)")

    plt.xlabel(r"Relative location $n$  ($\hat{\tau}-\tau=n$)")
    plt.ylabel("Probability")
    plt.title(r"Theory vs Empirical $\pi_n$ (T fixed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160); plt.show()
    print(f"[saved] {outpath}")


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

def zip_outputs(outdir, zipname=None, include_ext=None):
    """
    Zip all files in outdir into a single zip file.
    Returns the path to the zip file.

    Args:
        outdir: directory with outputs
        zipname: optional custom name (default = all_outputs.zip)
        include_ext: optional list of extensions (['.png','.csv']) to include only those
    """
    if zipname is None:
        zipname = os.path.join(outdir, "all_outputs.zip")
    else:
        zipname = os.path.join(outdir, zipname)

    with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(outdir):
            for file in files:
                if file == os.path.basename(zipname):
                    continue  # skip the zip itself
                if include_ext and not any(file.endswith(ext) for ext in include_ext):
                    continue  # skip unwanted extensions
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, outdir)
                zipf.write(file_path, arcname)

    print(f"[saved zip] {zipname}")
    return zipname


def main():
    # ----------- change inputs here -----------
    T       = 100
    tau     = 20
    theta0  = 0.60   # theta0 > theta1 (for Phi>0)
    theta1  = 0.50
    seed    = 50
    Ngrid   = T          # compute q,p on grid l+m <= Ngrid
    tieRule = "smallest"  # 'smallest' or 'largest'
    outdir  = "outputs_sec2_2"
    n_display = 10  # For pin distribution
    do_compare = True
    trials_for_comaprison = 2000  #MLE comparison trials for pin
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

    # 3) Random walks (Eqs. 2.6–2.7) and tau_hat via Eq. (2.13)
    t_star = tau_hat_from_Xt(Xt, tie_rule=tieRule)  # center at MLE
    left, right = build_walks_from_X(Xt, t_star, theta0, theta1)
    tau_hat_walk = tau_hat_from_walks(t_star, left, right)

    plot_detection_comparison(Xt, tau, tau_hat, tau_hat_walk, outdir)
    print("\n=== Estimation summary ===")
    print(f"true τ               : {tau}")
    print(f"τ̂ from X_t          : {tau_hat}")
    print(f"τ̂ from walks (2.13)  : {tau_hat_walk}")

    plot_walk(left["path"], left["M"], left["I"],
              "Left walk W", "walk_left.png", outdir)
    plot_walk(right["path"], right["M"], right["I"],
              "Right walk W'", "walk_right.png", outdir)

    # 4) §2.2 constants
    Phi = phi(theta0, theta1)
    p00  = p00_left(theta0, Phi)
    p00p = p00_right(theta1, Phi)
    print("\n=== §2.2 constants ===")
    print(f"Φ   = {Phi:.6f}")
    print(f"p00  = {p00:.6f} (Eq. 2.18)")
    print(f"p00' = {p00p:.6f} (Eq. 2.21)")

    # 5) Recursions for q,p (Eqs. 2.15–2.20)
    q_left,  p_left  = q_p_left(theta0, Phi, Ngrid, p00)
    q_right, p_right = q_p_right(theta1, Phi, Ngrid, p00p)

    # 6) Save PMFs + heatmaps
    df_p_left  = pmf_dict_to_df(p_left,  Phi, left_side=True)
    df_p_right = pmf_dict_to_df(p_right, Phi, left_side=False)
    save_df_csv(df_p_left,  os.path.join(outdir, "p_left_maxima.csv"))
    save_df_csv(df_p_right, os.path.join(outdir, "p_right_maxima.csv"))
    plot_heatmap_from_df(df_p_left,  "Heatmap p_{l,m} left",  os.path.join(outdir, "heatmap_p_left.png"))
    plot_heatmap_from_df(df_p_right, "Heatmap p'_{l,m} right", os.path.join(outdir, "heatmap_p_right.png"))
    print("\nMass checks:")
    print(f"Σ p_left  = {df_p_left['prob'].sum():.6f} + atom {p00:.6f}")
    print(f"Σ p_right = {df_p_right['prob'].sum():.6f} + atom {p00p:.6f}")

    # 7) §3 Asymptotic distribution pi_n (Eq. 3.1)
    
    pi = compute_pi_distribution(Phi, p_left, p_right, p00, p00p, n_display)
    print("\n=== π_n (Eq. 3.1) ===")
    for n in range(-n_display, n_display+1):
        print(f"{n:+d} : {pi.get(n, 0.0):.6f}")
    plot_pi_range(pi, n_display, os.path.join(outdir, "pi_distribution.png"))

    # --- Optional sanity-check comparison (simulation) ---
    if do_compare:
        trials = trials_for_comaprison
        emp_walk, emp_mle, tail_walk, tail_mle = empirical_pi_distributions(
            T, theta0, theta1, trials=trials, n_display=n_display, seed=12345
        )
        print(f"Empirical tail mass outside [-{n_display},{n_display}]: walk={tail_walk:.3f}, mle={tail_mle:.3f}")
        plot_theory_vs_empirical(
            pi_theory=pi,
            emp_walk=emp_walk,
            emp_mle=emp_mle,
            n_display=n_display,
            outpath=os.path.join(outdir, f"pi_theory_vs_empirical_T{T}_trials{trials}.png")
        )

    # 8) Zip everything
    zip_outputs(outdir, include_ext=[".png",".csv"])
    print(f"\nAll outputs saved and zipped in: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
