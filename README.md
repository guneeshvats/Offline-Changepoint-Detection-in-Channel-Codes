# Offline Changepoint Detection in Channel Codes

This repository contains the full implementation for **offline changepoint detection (CPD)** in **binary channel-coded communication streams** (BCH / LDPC) over BSC and AWGN channels.  
The core idea is a **two-stage pipeline**:

1. **Statistical Coarse MLE Stage**  
   Project each received codeword using a carefully selected vector \( h \in C_2^\perp \setminus C_1^\perp \).  
   This produces a Bernoulli sequence \( z_i = y_i \cdot h^T \).  
   A Hinkley-style Bernoulli MLE is used to get a coarse estimate \( \tilde{\tau} \) and a confidence window.

2. **Neural Refinement Stage**  
   A small 1D CNN scans raw codeword windows around the coarse region and refines the CP location.

This repository includes:
- MATLAB generation of channel-corrupted codewords  
- Python implementation of statistical CPD (MLE, CUSUM, PELT, BinSeg)  
- Deep learning notebooks for model training (known-p and unknown-p)  
- Final unified Python pipelines for **known-p** and **unknown-p** settings  

---

## Installation

```bash
git clone https://github.com/guneeshvats/Offline-Changepoint-Detection-in-Channel-Codes.git
cd Offline-Changepoint-Detection-in-Channel-Codes
pip install -r requirements.txt
```

## Directory Structure 
```bash
Offline-Changepoint-Detection-in-Channel-Codes/
│
├── Data_Generation/                         # MATLAB scripts for creating simulated codeword pools
│   ├── BCH+AWGN_Channel/                    # BCH over AWGN (generates CSV pools)
│   ├── BCH+BSC_Channel/                     # BCH over BSC (generates CSV pools)
│   ├── LDPC+AWGN_Channel/                   # LDPC over AWGN
│   └── LDPC+BSC_Channel/                    # LDPC over BSC
│
├── Deep_Learning_Approach/                  # Jupyter notebooks for training & evaluation
│   ├── BCH+BSC_known_p/                     # Known–p models
│   │   ├── n=15/                            # Contains T={5,7,10} trained models + experiments
│   │   └── n=31/                            # Contains T={5,7,10} trained models + experiments
│   └── BCH+BSC_unknown_p/                   # Unknown–p Type-II models (trained on multi-p data)
│       ├── n=15/
│       └── n=31/
│
├── Hinkley_Implementation/                  # Pure statistical CPD (Bernoulli MLE + πₙ recursion)
│   ├── code.py                              # Implements MLE, recursive πₙ, CUSUM, etc.
│   └── README.md
│
├── Processed_Data/
│   ├── Bernoulli_Data/                      # Ruptures baselines: BinSeg, PELT, CUSUM, MLE
│   └── PMF1_PMF2/                           # Probability mass visualizations and utilities
│
├── final_algorithm_known_p.py               # ***Main unified algorithm (known BSC-p)***
├── final_algorithm_unknown_p.py             # Unified pipeline for unknown-p inference
│
├── Find_min_weight_h_BCH_n15_n31.m          # MATLAB search for low-weight h (BCH)
├── Find_min_weight_h_LDPC_n648.m            # MATLAB search for low-weight h (LDPC)
│
├── requirements.txt
└── README.md
```

## Quick Guide

1. To run the final algorithm → use : final_algorithm_known_p.py or final_algorithm_unknown_p.py
2. To train CNN models → open notebooks under Deep_Learning_Approach
3. To regenerate data → run scripts under Data_Generation
4. To study the math → read Hinkley_Implementation/code.py


### Running the Final Algorithm (Known-p)

The script final_algorithm_known_p.py implements the entire pipeline:

1. Load C1/C2 channel-corrupted codeword pools
2. Build a sequence of length M
3. Project using h → Bernoulli sequence z
4. Run coarse MLE + compute window using πₙ
5. Run a TorchScript CNN for refinement
6. Produce final estimate τ^


### 1. Configure Inputs (Inside the Script)

Open final_algorithm_known_p.py and edit the block:
```python
# ========================= USER INPUTS ===========================

# 1) Channel codeword pools (CSV files with rows of 0/1)
CSV_C1 = "/path/to/bsc_p0.005_C1_n15_k11.csv"
CSV_C2 = "/path/to/bsc_p0.005_C2_n15_k7.csv"

# 2) Global sequence length and true CP (for synthetic experiments)
M = 1000
TAU_TRUE = 500

# 3) Binary test vector h (length n)
H = [1,1,0,0,0,0,0,0,0,1,0,0,0,1,0]   # Example for n=15

# 4) BSC parameters
p = 0.005
SWAP_THETA12 = True

# 5) Confidence level for selecting π_n window
alpha = 0.90

# 6) Neural model (TorchScript)
MODEL_PATH = "/path/to/models/n15_T5_p0005.pt"
WINDOW_T  = 5
STRIDE    = 1
DEVICE    = "cpu"

# 7) Random seed
SEED = 12345

# 8) π_n recursion hyperparameters
PIN_NGRID    = 20
PIN_NDISPLAY = 20
VIEW_HALF    = 10

# 9) Monte-Carlo experiments (optional)
MC_ENABLE = False
MC_NR = 3000
```
### 2. Run the Script

```bash
python final_algorithm_known_p.py
```

### Example output : 

```
================= RESULT =================
True changepoint τ (count) : 500
Coarse MLE τ~ (on z_i)    : 492
α-window S (even), half   : S=20, halfwidth=10
Refined NN τ̂ (count)     : 498
==========================================
[saved] result_final_algorithm.csv
```

### Understanding the Required Inputs

The parameters in the `USER INPUTS` section correspond directly to the components of the CPD pipeline and must be provided by the user before running the algorithm:

- **Channel Codeword Pools (CSV_C1, CSV_C2)**  
  These CSV files contain rows of binary codewords (0/1) generated from the MATLAB scripts under `Data_Generation/`.  
  For a given `(n, k)` BCH or LDPC code and a chosen channel (BSC or AWGN), run the appropriate `.m` script to create the CSV pools.  
  C1 corresponds to the code used *before* the changepoint and C2 corresponds to the code *after* the changepoint.

- **Sequence Length (M) and True Changepoint (TAU_TRUE)**  
  These decide how long the synthetic sequence is and where the true changepoint occurs.  
  If you want to run the algorithm on **a single custom sequence**, set `MC_ENABLE = False` and simply choose any value of `TAU_TRUE` (it is used only for reporting).  
  If you want **Monte-Carlo experiments**, set `MC_ENABLE = True` and specify how many random trials using `MC_NR`.

- **Projection Vector h**  
  This is a binary vector of length `n` that must satisfy  
  \( h \in C_2^\perp \setminus C_1^\perp \).  
  For BCH (n = 15, 31) or LDPC (n = 648), use the MATLAB scripts  
  `Find_min_weight_h_BCH_n15_n31.m` or `Find_min_weight_h_LDPC_n648.m`  
  to search for a **low-weight** valid `h`.  
  A well-chosen `h` strongly improves the statistical discriminability in the Bernoulli projection.

- **Neural Model Path (MODEL_PATH)**  
  The neural refinement stage uses a TorchScript `.pt` file trained on the appropriate code length `n`, window size `T`, and (for known-p) the BSC parameter `p`.  
  To obtain this file, open the notebooks inside `Deep_Learning_Approach/`, run the training cells, and export the trained model.  
  The `.pt` file from the notebook is exactly what you provide here.

- **BSC Parameter p (Known-p Pipeline Only)**  
  This is the true BSC crossover probability used both in data generation and in computing the Bernoulli parameters \( \theta_1, \theta_2 \).  
  For the unknown-p model, **this parameter is not required** at inference.

- **Confidence Parameter α**  
  α (typically 0.90 or 0.95) determines how much probability mass of the πₙ distribution to keep when selecting the refinement window around the coarse MLE estimate.  
  Larger α → larger window → more context for the neural network.

- **πₙ Recursion Hyperparameters (PIN_NGRID, PIN_NDISPLAY, VIEW_HALF)**  
  These influence the numerical resolution of the lattice used in computing the Hinkley recursion πₙ.  
  Users typically do not change these unless experimenting with the statistical stage.

Together, these inputs fully define the pipeline: **data source → statistical coarse estimate → πₙ window → neural refinement**.





### Output Files

`result_final_algorithm.csv` contains:

1. All input parameters
2. θ₁, θ₂, coarse estimate
3. Window width
4. Crop indices
5. Final estimate
6. πₙ distribution values


### Unknown-p Pipeline

```bash
python final_algorithm_unknown_p.py
```
This uses a Type-II CNN model trained across multiple p values.
No BSC parameter p is needed at inference.


## Hinkley Statistical Baselines

The folder: `Hinkley_Implementation/`
contains:
code.py → exact MLE recursion, πₙ computation, CUSUM, and baseline methods
README.md → notes explaining the maths
Useful for benchmarking MLE-only CPD vs. neural refinement.



